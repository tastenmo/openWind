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

Stage 1 loads `data/MouthpieceDN16V2.json` by default, uses its 6.5 mm exit
radius as the constant main-pipe bore, and optimizes only pipe length and
terminal radius against the key's root and octave resonances. A linear taper to
the optimized end radius occupies the final 20 mm. Pass
`starting_bore_radius_m` to `run_stage_1_length_endhole_optimization` to choose
a different main-pipe radius. The multi-key notebook uses DN16 PVC with a
14.2 mm inner diameter (7.1 mm radius); the calibrated mouthpiece stays unchanged.

Stage 1 provides a provisional length for the unperforated pipe. Stage 2 jointly
optimizes tonehole positions, diameters and pipe length (within +/-5% of Stage 1),
accounting for the acoustic loading of closed tonehole cavities. Bore radii and
the final 20 mm taper remain unchanged. The all-closed fundamental is constrained
to within 0.1 cent of its tuning target in the optimization sweep, rather than
traded off against the other notes.
Its acoustic simulation includes the
same mouthpiece geometry and offsets hole positions into the combined instrument
coordinate system. After Stage 2, use `pipeline.tonehole_result.bore` for tables,
simulation and Stage 3/export, rather than the provisional Stage 1 bore.

The multi-key notebook displays a millimeter drilling table (positions relative
to the pipe start) and separate fingering/tuning tables for both registers.
Each register is simulated independently, including the fork and the vented
upper root. Signed cent errors use the selected temperament; missing resonance
peaks are marked explicitly. This evaluation does not add upper-register notes
to the optimization objective.
