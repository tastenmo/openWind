# High D Tin Whistle OpenWind

Poetry project for simulating and optimizing a tapered High D tin whistle.

The baseline model uses OpenWind's supported `InstrumentGeometry` and frequency-domain impedance solver. OpenWind 0.12.4 accepts `radius_out` for linear conical side holes, so the configured inner and outer hole diameters are passed directly into the acoustic model as well as reported for manufacturing.

## Setup

```powershell
poetry install
poetry run python -m ipykernel install --user --name tin-whistle-openwind --display-name "Tin Whistle OpenWind"
poetry run pytest
```

Open `notebooks/01_geometry_and_fingering.ipynb` first. Replace the placeholder `L1` through `L6` positions in `data/design_defaults.yaml` before using optimization results for a physical instrument.

## Layout

- `src/tin_whistle_openwind/`: reusable geometry, fingering, acoustics, optimization, and undercutting code
- `data/`: design inputs and target frequencies
- `notebooks/`: staged experiments
- `artifacts/`: generated figures and tables
- `tests/`: focused regression tests
