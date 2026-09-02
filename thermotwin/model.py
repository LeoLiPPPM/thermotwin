"""Coupled thermal, solar-generation, and battery dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class ThermalParameters:
    """Model constants using kW, kJ, seconds, and degrees Celsius."""

    air_capacity_kj_k: float = 1600.0
    wall_capacity_kj_k: float = 900.0
    air_wall_resistance_k_kw: float = 3.00
    wall_ambient_resistance_k_kw: float = 65.0
    solar_gain_kw_per_w_m2: float = 0.00005
    internal_heat_kw: float = 0.018
    cooling_cop: float = 1.55
    max_cooling_power_kw: float = 0.36
    base_electrical_load_kw: float = 0.025
    pv_rated_power_kw: float = 0.55
    battery_capacity_kwh: float = 1.60
    charge_efficiency: float = 0.94
    discharge_efficiency: float = 0.94

    def __post_init__(self) -> None:
        positive = (
            self.air_capacity_kj_k,
            self.wall_capacity_kj_k,
            self.air_wall_resistance_k_kw,
            self.wall_ambient_resistance_k_kw,
            self.cooling_cop,
            self.max_cooling_power_kw,
            self.pv_rated_power_kw,
            self.battery_capacity_kwh,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("thermal and electrical scale parameters must be positive")


@dataclass(frozen=True, slots=True)
class Environment:
    ambient_temperature_c: float
    irradiance_w_m2: float


@dataclass(frozen=True, slots=True)
class CoolerState:
    air_temperature_c: float
    wall_temperature_c: float
    battery_energy_kwh: float


@dataclass(frozen=True, slots=True)
class StepOutcome:
    state: CoolerState
    pv_power_kw: float
    cooling_power_kw: float
    backup_energy_kwh: float
    curtailed_solar_energy_kwh: float


def thermal_derivative(
    air_temperature_c: float,
    wall_temperature_c: float,
    environment: Environment,
    cooling_power_kw: float,
    parameters: ThermalParameters,
    *,
    extra_heat_kw: float = 0.0,
) -> tuple[float, float]:
    """Return ``dT_air/dt`` and ``dT_wall/dt`` in degrees C per second."""

    air_wall_heat_kw = (wall_temperature_c - air_temperature_c) / parameters.air_wall_resistance_k_kw
    ambient_wall_heat_kw = (
        environment.ambient_temperature_c - wall_temperature_c
    ) / parameters.wall_ambient_resistance_k_kw
    solar_heat_kw = parameters.solar_gain_kw_per_w_m2 * max(environment.irradiance_w_m2, 0.0)
    cooling_heat_kw = parameters.cooling_cop * cooling_power_kw
    air_rate = (air_wall_heat_kw + parameters.internal_heat_kw + extra_heat_kw - cooling_heat_kw) / parameters.air_capacity_kj_k
    wall_rate = (ambient_wall_heat_kw - air_wall_heat_kw + solar_heat_kw) / parameters.wall_capacity_kj_k
    return air_rate, wall_rate


def _thermal_rk4(
    state: CoolerState,
    environment: Environment,
    cooling_power_kw: float,
    duration_s: float,
    parameters: ThermalParameters,
    extra_heat_kw: float,
) -> tuple[float, float]:
    def derivative(air: float, wall: float) -> np.ndarray:
        return np.asarray(
            thermal_derivative(
                air,
                wall,
                environment,
                cooling_power_kw,
                parameters,
                extra_heat_kw=extra_heat_kw,
            )
        )

    initial = np.array([state.air_temperature_c, state.wall_temperature_c], dtype=float)
    k1 = derivative(*initial)
    k2 = derivative(*(initial + 0.5 * duration_s * k1))
    k3 = derivative(*(initial + 0.5 * duration_s * k2))
    k4 = derivative(*(initial + duration_s * k3))
    final = initial + duration_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return float(final[0]), float(final[1])


def step(
    state: CoolerState,
    environment: Environment,
    commanded_cooling_power_kw: float,
    duration_s: float,
    parameters: ThermalParameters,
    *,
    extra_heat_kw: float = 0.0,
) -> StepOutcome:
    """Advance the digital twin one control interval.

    A backup source supplies any demand that PV and the battery cannot meet.
    The reported backup energy is therefore a resilience metric and the
    commanded cooling power remains physically available.
    """

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    cooling_power_kw = float(np.clip(commanded_cooling_power_kw, 0.0, parameters.max_cooling_power_kw))
    pv_power_kw = parameters.pv_rated_power_kw * np.clip(environment.irradiance_w_m2 / 1000.0, 0.0, 1.25)
    demand_kw = parameters.base_electrical_load_kw + cooling_power_kw
    duration_h = duration_s / 3600.0
    battery = float(np.clip(state.battery_energy_kwh, 0.0, parameters.battery_capacity_kwh))
    backup_energy = 0.0
    curtailed_energy = 0.0

    if pv_power_kw >= demand_kw:
        available_charge = (pv_power_kw - demand_kw) * duration_h * parameters.charge_efficiency
        accepted_charge = min(available_charge, parameters.battery_capacity_kwh - battery)
        battery += accepted_charge
        curtailed_energy = max(0.0, available_charge - accepted_charge) / parameters.charge_efficiency
    else:
        deficit_energy = (demand_kw - pv_power_kw) * duration_h
        deliverable_battery_energy = battery * parameters.discharge_efficiency
        supplied = min(deficit_energy, deliverable_battery_energy)
        battery -= supplied / parameters.discharge_efficiency
        backup_energy = deficit_energy - supplied

    air, wall = _thermal_rk4(
        state,
        environment,
        cooling_power_kw,
        duration_s,
        parameters,
        extra_heat_kw,
    )
    next_state = replace(
        state,
        air_temperature_c=air,
        wall_temperature_c=wall,
        battery_energy_kwh=float(np.clip(battery, 0.0, parameters.battery_capacity_kwh)),
    )
    return StepOutcome(next_state, float(pv_power_kw), cooling_power_kw, backup_energy, curtailed_energy)
