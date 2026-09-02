"""Calibrate, control, benchmark, and diagnose the solar cooler digital twin."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .calibration import fit_thermal_parameters, synthetic_calibration_data
from .controllers import ModelPredictiveController, Thermostat
from .model import CoolerState, Environment, ThermalParameters, step


@dataclass(frozen=True, slots=True)
class Run:
    air_c: np.ndarray
    wall_c: np.ndarray
    battery_kwh: np.ndarray
    cooling_kw: np.ndarray
    pv_kw: np.ndarray
    backup_kwh: np.ndarray


def make_day(steps: int, interval_s: float) -> list[Environment]:
    time_h = np.arange(steps) * interval_s / 3600.0
    ambient = 28.0 + 7.0 * np.sin(2.0 * np.pi * (time_h - 8.0) / 24.0)
    solar = 920.0 * np.maximum(0.0, np.sin(np.pi * (time_h - 6.0) / 12.0))
    return [Environment(float(t), float(g)) for t, g in zip(ambient, solar, strict=True)]


def simulate_controller(
    environments: list[Environment],
    parameters: ThermalParameters,
    controller: Thermostat | ModelPredictiveController,
    interval_s: float,
    *,
    injected_fault: bool = False,
) -> Run:
    state = CoolerState(4.5, 8.0, 0.72 * parameters.battery_capacity_kwh)
    air = [state.air_temperature_c]
    wall = [state.wall_temperature_c]
    battery = [state.battery_energy_kwh]
    cooling: list[float] = []
    pv: list[float] = []
    backup: list[float] = []

    for index, conditions in enumerate(environments):
        if isinstance(controller, Thermostat):
            action = controller.choose(state, parameters)
        else:
            action = controller.choose(state, environments[index : index + controller.horizon_steps])
        time_h = index * interval_s / 3600.0
        extra_heat = 0.22 if injected_fault and 14.0 <= time_h < 15.0 else 0.0
        outcome = step(state, conditions, action, interval_s, parameters, extra_heat_kw=extra_heat)
        state = outcome.state
        air.append(state.air_temperature_c)
        wall.append(state.wall_temperature_c)
        battery.append(state.battery_energy_kwh)
        cooling.append(outcome.cooling_power_kw)
        pv.append(outcome.pv_power_kw)
        backup.append(outcome.backup_energy_kwh)
    return Run(*(np.asarray(values) for values in (air, wall, battery, cooling, pv, backup)))


def metrics(run: Run, interval_s: float, battery_capacity_kwh: float) -> dict[str, float]:
    temperature = run.air_c[1:]
    outside = (temperature < 2.0) | (temperature > 8.0)
    switching = np.count_nonzero(np.diff(run.cooling_kw) != 0.0)
    return {
        "backup_energy_wh": float(1000.0 * np.sum(run.backup_kwh)),
        "cooling_energy_wh": float(1000.0 * np.sum(run.cooling_kw) * interval_s / 3600.0),
        "temperature_violation_minutes": float(np.sum(outside) * interval_s / 60.0),
        "minimum_temperature_c": float(np.min(temperature)),
        "maximum_temperature_c": float(np.max(temperature)),
        "compressor_action_changes": float(switching),
        "ending_battery_percent": float(100.0 * run.battery_kwh[-1] / battery_capacity_kwh),
    }


def fault_residuals(
    observed: Run,
    environments: list[Environment],
    parameters: ThermalParameters,
    interval_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    for index, environment in enumerate(environments):
        state = CoolerState(observed.air_c[index], observed.wall_c[index], observed.battery_kwh[index])
        predicted = step(state, environment, observed.cooling_kw[index], interval_s, parameters).state
        predictions.append(predicted.air_temperature_c)
    residual = observed.air_c[1:] - np.asarray(predictions)
    baseline = residual[: int(12.0 * 3600.0 / interval_s)]
    robust_sigma = max(1.4826 * np.median(np.abs(baseline - np.median(baseline))), 1e-4)
    alarm = residual > np.median(baseline) + 4.0 * robust_sigma
    return residual, alarm


def run_demo(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    truth = ThermalParameters()
    calibration_data = synthetic_calibration_data(truth)
    deliberately_biased = replace(
        truth,
        air_wall_resistance_k_kw=4.50,
        wall_ambient_resistance_k_kw=42.0,
        solar_gain_kw_per_w_m2=0.00009,
    )
    calibration = fit_thermal_parameters(calibration_data, deliberately_biased)

    interval_s = 900.0
    environments = make_day(96, interval_s)
    thermostat_run = simulate_controller(environments, calibration.parameters, Thermostat(), interval_s)
    mpc_run = simulate_controller(
        environments,
        calibration.parameters,
        ModelPredictiveController(calibration.parameters, horizon_steps=6, interval_s=interval_s),
        interval_s,
    )
    fault_run = simulate_controller(
        environments,
        calibration.parameters,
        ModelPredictiveController(calibration.parameters, horizon_steps=6, interval_s=interval_s),
        interval_s,
        injected_fault=True,
    )
    residual, alarm = fault_residuals(fault_run, environments, calibration.parameters, interval_s)
    alarm_indices = np.flatnonzero(alarm)

    summary: dict[str, object] = {
        "calibration": {
            "air_rmse_c": calibration.air_rmse_c,
            "wall_rmse_c": calibration.wall_rmse_c,
            "evaluations": calibration.evaluations,
            "fitted_air_wall_resistance_k_kw": calibration.parameters.air_wall_resistance_k_kw,
            "fitted_wall_ambient_resistance_k_kw": calibration.parameters.wall_ambient_resistance_k_kw,
            "fitted_solar_gain_kw_per_w_m2": calibration.parameters.solar_gain_kw_per_w_m2,
        },
        "thermostat": metrics(thermostat_run, interval_s, calibration.parameters.battery_capacity_kwh),
        "mpc": metrics(mpc_run, interval_s, calibration.parameters.battery_capacity_kwh),
        "fault_detection": {
            "first_alarm_hour": None if alarm_indices.size == 0 else float(alarm_indices[0] * interval_s / 3600.0),
            "alarm_intervals": int(np.count_nonzero(alarm)),
        },
        "benchmark_scope": "synthetic simulation; not a hardware result",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    time_state_h = np.arange(97) * interval_s / 3600.0
    time_step_h = np.arange(96) * interval_s / 3600.0
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time_state_h, thermostat_run.air_c, label="thermostat", color="#9a9a9a")
    axes[0].plot(time_state_h, mpc_run.air_c, label="predictive control", color="#315f8c")
    axes[0].axhspan(2.0, 8.0, color="#b8d8ba", alpha=0.25, label="safe band")
    axes[0].set_ylabel("air temp [C]")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].set_title("Solar cooler digital-twin benchmark")

    axes[1].step(time_step_h, mpc_run.cooling_kw, where="post", label="cooling demand", color="#315f8c")
    axes[1].plot(time_step_h, mpc_run.pv_kw, label="PV generation", color="#e09f3e")
    axes[1].set_ylabel("power [kW]")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    axes[2].plot(time_state_h, 100.0 * mpc_run.battery_kwh / calibration.parameters.battery_capacity_kwh, color="#557a46")
    axes[2].set_ylabel("battery [%]")
    axes[2].set_ylim(-2.0, 102.0)

    axes[3].plot(time_step_h, residual, color="#315f8c", label="one-step residual")
    axes[3].scatter(time_step_h[alarm], residual[alarm], color="#d14b40", s=18, label="fault alarm")
    axes[3].axvspan(14.0, 15.0, color="#d14b40", alpha=0.12, label="injected door fault")
    axes[3].set_xlabel("time [hour]")
    axes[3].set_ylabel("residual [C]")
    axes[3].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "digital_twin_benchmark.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("thermotwin/outputs"))
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output), indent=2))


if __name__ == "__main__":
    main()
