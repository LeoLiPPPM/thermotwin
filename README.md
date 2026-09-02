# ThermoTwin

ThermoTwin turns the SVcooler idea into a physics-informed software project. It
contains a two-node thermal model, a photovoltaic and battery model, synthetic
sensor-data calibration, an enumerative model-predictive controller, and
residual-based fault detection.

## Run

```bash
PYTHONPATH=thermotwin python -m thermotwin.demo --output thermotwin/outputs
python -m unittest discover -s thermotwin/tests -v
```

The demo compares a thermostat against predictive control under exactly the
same ambient temperature and solar-irradiance profile. Its generated numbers
are simulation benchmarks, not hardware results. Replace the synthetic data
with logged SVcooler sensor data before describing any result as experimental.

