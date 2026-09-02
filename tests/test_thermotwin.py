from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermotwin.calibration import fit_thermal_parameters, synthetic_calibration_data
from thermotwin.model import CoolerState, Environment, ThermalParameters, step, thermal_derivative


class ThermoTwinTests(unittest.TestCase):
    def test_equilibrium_has_zero_derivative(self) -> None:
        parameters = replace(ThermalParameters(), internal_heat_kw=0.0, solar_gain_kw_per_w_m2=0.0)
        rates = thermal_derivative(10.0, 10.0, Environment(10.0, 0.0), 0.0, parameters)
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[1], 0.0)

    def test_battery_bounds_are_enforced(self) -> None:
        parameters = ThermalParameters()
        state = CoolerState(5.0, 8.0, parameters.battery_capacity_kwh)
        charged = step(state, Environment(25.0, 1200.0), 0.0, 3600.0, parameters)
        self.assertLessEqual(charged.state.battery_energy_kwh, parameters.battery_capacity_kwh)
        empty = CoolerState(5.0, 8.0, 0.0)
        discharged = step(empty, Environment(25.0, 0.0), parameters.max_cooling_power_kw, 3600.0, parameters)
        self.assertGreaterEqual(discharged.state.battery_energy_kwh, 0.0)
        self.assertGreater(discharged.backup_energy_kwh, 0.0)

    def test_synthetic_calibration_recovers_parameters(self) -> None:
        truth = ThermalParameters()
        data = synthetic_calibration_data(truth, seed=7)
        initial = replace(
            truth,
            air_wall_resistance_k_kw=4.80,
            wall_ambient_resistance_k_kw=45.0,
            solar_gain_kw_per_w_m2=0.00010,
        )
        fitted = fit_thermal_parameters(data, initial)
        self.assertLess(abs(fitted.parameters.air_wall_resistance_k_kw / truth.air_wall_resistance_k_kw - 1.0), 0.04)
        self.assertLess(abs(fitted.parameters.wall_ambient_resistance_k_kw / truth.wall_ambient_resistance_k_kw - 1.0), 0.04)
        self.assertLess(abs(fitted.parameters.solar_gain_kw_per_w_m2 / truth.solar_gain_kw_per_w_m2 - 1.0), 0.12)


if __name__ == "__main__":
    unittest.main()
