"""Fit physically meaningful thermal parameters to sensor histories."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .model import CoolerState, Environment, ThermalParameters, step


@dataclass(frozen=True, slots=True)
class CalibrationData:
    environments: tuple[Environment, ...]
    cooling_power_kw: NDArray[np.float64]
    observed_air_c: NDArray[np.float64]
    observed_wall_c: NDArray[np.float64]
    interval_s: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    parameters: ThermalParameters
    air_rmse_c: float
    wall_rmse_c: float
    evaluations: int


def _temperature_history(
    initial_air_c: float,
    initial_wall_c: float,
    data: CalibrationData,
    parameters: ThermalParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    state = CoolerState(initial_air_c, initial_wall_c, 0.5 * parameters.battery_capacity_kwh)
    air = [initial_air_c]
    wall = [initial_wall_c]
    for environment, power in zip(data.environments, data.cooling_power_kw, strict=True):
        state = step(state, environment, float(power), data.interval_s, parameters).state
        air.append(state.air_temperature_c)
        wall.append(state.wall_temperature_c)
    return np.asarray(air), np.asarray(wall)


def synthetic_calibration_data(
    truth: ThermalParameters,
    *,
    seed: int = 41,
    interval_s: float = 600.0,
    steps: int = 72,
) -> CalibrationData:
    """Generate a deterministic commissioning experiment for the demo."""

    rng = np.random.default_rng(seed)
    time_h = np.arange(steps) * interval_s / 3600.0
    ambient = 25.0 + 4.5 * np.sin(2.0 * np.pi * (time_h - 2.0) / 24.0)
    irradiance = 850.0 * np.maximum(0.0, np.sin(np.pi * (time_h + 1.0) / 12.0))
    environments = tuple(Environment(float(t), float(g)) for t, g in zip(ambient, irradiance, strict=True))
    blocks = rng.integers(0, 3, size=(steps + 2) // 3)
    power = np.repeat(blocks, 3)[:steps] * (0.5 * truth.max_cooling_power_kw)

    placeholder = CalibrationData(environments, power, np.empty(0), np.empty(0), interval_s)
    air, wall = _temperature_history(5.0, 8.0, placeholder, truth)
    return CalibrationData(
        environments,
        power,
        air + rng.normal(0.0, 0.035, size=air.size),
        wall + rng.normal(0.0, 0.035, size=wall.size),
        interval_s,
    )


def fit_thermal_parameters(data: CalibrationData, initial: ThermalParameters) -> CalibrationResult:
    """Estimate two thermal resistances and the solar-gain coefficient."""

    initial_vector = np.array(
        [
            initial.air_wall_resistance_k_kw,
            initial.wall_ambient_resistance_k_kw,
            initial.solar_gain_kw_per_w_m2,
        ]
    )

    def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        candidate = replace(
            initial,
            air_wall_resistance_k_kw=float(vector[0]),
            wall_ambient_resistance_k_kw=float(vector[1]),
            solar_gain_kw_per_w_m2=float(vector[2]),
        )
        air, wall = _temperature_history(
            float(data.observed_air_c[0]),
            float(data.observed_wall_c[0]),
            data,
            candidate,
        )
        return np.concatenate((air - data.observed_air_c, wall - data.observed_wall_c))

    solution = least_squares(
        residual,
        initial_vector,
        bounds=([1.00, 20.0, 0.000005], [10.0, 140.0, 0.00030]),
        x_scale="jac",
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    fitted = replace(
        initial,
        air_wall_resistance_k_kw=float(solution.x[0]),
        wall_ambient_resistance_k_kw=float(solution.x[1]),
        solar_gain_kw_per_w_m2=float(solution.x[2]),
    )
    fitted_air, fitted_wall = _temperature_history(
        float(data.observed_air_c[0]),
        float(data.observed_wall_c[0]),
        data,
        fitted,
    )
    return CalibrationResult(
        fitted,
        float(np.sqrt(np.mean((fitted_air - data.observed_air_c) ** 2))),
        float(np.sqrt(np.mean((fitted_wall - data.observed_wall_c) ** 2))),
        int(solution.nfev),
    )
