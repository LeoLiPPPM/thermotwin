"""Baseline and receding-horizon temperature controllers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .model import CoolerState, Environment, ThermalParameters, step


@dataclass(slots=True)
class Thermostat:
    """Hysteresis controller used as an interpretable baseline."""

    turn_on_c: float = 5.0
    turn_off_c: float = 3.5
    is_on: bool = False

    def choose(self, state: CoolerState, parameters: ThermalParameters) -> float:
        if state.air_temperature_c >= self.turn_on_c:
            self.is_on = True
        elif state.air_temperature_c <= self.turn_off_c:
            self.is_on = False
        return parameters.max_cooling_power_kw if self.is_on else 0.0


class ModelPredictiveController:
    """Finite-action MPC solved by exhaustive search over a short horizon."""

    def __init__(
        self,
        parameters: ThermalParameters,
        *,
        horizon_steps: int = 6,
        interval_s: float = 900.0,
    ) -> None:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        self.parameters = parameters
        self.horizon_steps = horizon_steps
        self.interval_s = interval_s
        self.actions = (0.0, 0.5 * parameters.max_cooling_power_kw, parameters.max_cooling_power_kw)
        self.previous_action = 0.0
        self._sequences = tuple(product(self.actions, repeat=horizon_steps))

    def choose(self, state: CoolerState, forecast: list[Environment]) -> float:
        if not forecast:
            raise ValueError("MPC requires at least one forecast step")
        environment = forecast[: self.horizon_steps]
        if len(environment) < self.horizon_steps:
            environment.extend([environment[-1]] * (self.horizon_steps - len(environment)))

        best_cost = np.inf
        best_action = self.actions[0]
        for actions in self._sequences:
            candidate = state
            previous = self.previous_action
            cost = 0.0
            for action, conditions in zip(actions, environment, strict=True):
                outcome = step(candidate, conditions, action, self.interval_s, self.parameters)
                candidate = outcome.state
                temperature = candidate.air_temperature_c
                violation = max(2.0 - temperature, 0.0) + max(temperature - 8.0, 0.0)
                tracking = temperature - 4.2
                reserve_fraction = candidate.battery_energy_kwh / self.parameters.battery_capacity_kwh
                cost += 140.0 * violation**2 + 0.28 * tracking**2
                cost += 45.0 * outcome.backup_energy_kwh + 0.10 * (1.0 - reserve_fraction) ** 2
                cost += 0.25 * abs(action - previous) / self.parameters.max_cooling_power_kw
                previous = action
            if cost < best_cost:
                best_cost = cost
                best_action = actions[0]
        self.previous_action = best_action
        return best_action

