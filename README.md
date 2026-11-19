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
- `notebooks/`: staged experiments, including `genetic_geometry_optimization.ipynb` for defining, running, and analyzing the geometry GA
- `artifacts/`: generated figures and tables
- `tests/`: focused regression tests

The pipeline loads `data/MouthpieceDN16V2.json` by default; pass
`mouthpiece_name="MouthpieceDN16V4"` to `WhistleDesignPipeline` to select the
alternate calibrated geometry. The selected mouthpiece is used consistently
in both optimization stages and recorded in the exported design metadata.
Stage 1 uses the mouthpiece's 6.5 mm exit radius as the constant main-pipe bore
unless a starting radius is supplied. By default it optimizes pipe length and
terminal radius against the key's root and octave resonances, with a linear
20 mm end taper. Set `optimize_endhole=False` on `WhistleDesignPipeline` for a
straight PVC bore; then Stage 1 optimizes length only and keeps the pipe radius
constant through its open end. Pass
`starting_bore_radius_m` to `run_stage_1_length_endhole_optimization` to choose
a different main-pipe radius. The multi-key notebook uses DN16 PVC with a
14.2 mm inner diameter (7.1 mm radius); the calibrated mouthpiece stays unchanged.

Stage 1 provides a provisional length for the unperforated pipe. Stage 2 jointly
optimizes tonehole positions, diameters and pipe length (within +/-5% of Stage 1),
accounting for the acoustic loading of closed tonehole cavities. The Stage 1 bore
profile remains unchanged. The all-closed fundamental is constrained
to within 0.1 cent of its tuning target in the optimization sweep, rather than
traded off against the other notes.
Its acoustic simulation includes the
same mouthpiece geometry and offsets hole positions into the combined instrument
coordinate system. After Stage 2, use `pipeline.tonehole_result.bore` for tables,
simulation and Stage 3/export, rather than the provisional Stage 1 bore.

An optional genetic refinement can search the Stage 2 design jointly. Its
20-gene geometry DNA contains pipe length, six tonehole positions, six bore
radii evaluated at those moving hole positions, six hole diameters, and one
shared chimney bevel angle. The bore radii are limited to a configurable
fraction around the seed profile (5% by default); the decoded profile
interpolates those local radius changes through the seed bore while retaining
its start and end treatment. The search also respects the existing ±5% pipe
length range, 5–8 mm hole diameters, within-hand spacing, hole order,
30 mm bottom-hole clearance, and 0–15° bevel bounds.

```python
from tin_whistle_openwind import GeneticAlgorithmConfig, WhistleDesignPipeline

pipeline = WhistleDesignPipeline("Soprano_D", optimize_endhole=False)
bore = pipeline.run_stage_1_length_endhole_optimization()
holes = pipeline.run_stage_2_tonehole_optimization(bore)
seed_bore = pipeline.tonehole_result.bore
result = pipeline.run_genetic_refinement(
    seed_bore,
    holes,
    config=GeneticAlgorithmConfig(
        population_size=20,
        generations=10,
        random_seed=42,
        include_second_register=True,
        optimize_endhole=True,
        optimize_bevel=False,
    ),
)
design = pipeline.run_stage_3_undercut_optimization(result.bore, result.layout)
```

This example starts with a cylindrical seed and lets the GA vary the six
hole positions, diameters, local bore radii, and the endhole radius in one
search. With `notes=None`, the GA scores eight lower-register targets
(including the fork) and seven upper-register targets (no forks).
The upper notes use their standard fingerings at twice the lower
target, except for the root, which uses the vented fingering. The endhole
radius uses the same relative bound as the other bore radii, and the GA
adds a 20 mm terminal taper when the seed is cylindrical. Bevel remains
unchanged. See the [executed notebook](notebooks/genetic_geometry_optimization.ipynb)
for the before-and-after tuning by register.

The NumPy-based genetic search is opt-in and does not change the existing
staged optimizers. When `Degree_1` is selected, it retunes each candidate's
pipe length within the existing geometry limits before scoring the other
notes. Candidates meeting the 0.1-cent closed-root tolerance rank ahead of
those that miss it; if none meets it, the result reports the closest root
and marks the tolerance as unmet. Use `retune_closed_root=False` to compare
against the original penalty-based search. The result reports the best
geometry, before/after tuning errors and costs, and the number of unique
candidates scored (not the number of acoustic sweeps). A fixed `random_seed`
makes the evolutionary operations reproducible.

The 18 mm minimum tonehole spacing applies within each hand only (L1-L2,
L2-L3, R1-R2, R2-R3), including measured-impedance inversion. There is no
18 mm minimum between L3 and R1; optimization retains their axial order with
a 0.1 mm numerical clearance. This is not a drill-edge clearance constraint.

The multi-key notebook displays a millimeter drilling table (positions relative
to the pipe start) and separate fingering/tuning tables for both registers.
Each register is simulated independently, including the fork and the vented
upper root. Signed cent errors use the selected temperament; missing resonance
peaks are marked explicitly. This evaluation does not add upper-register notes
to the optimization objective.

Section 5 of the multi-key notebook compares actual impedance and admittance
spectra for all nine standard fingerings (including fork and vented upper root),
using the final design and calibrated mouthpiece. All interior peaks in the
displayed frequency range are tabulated with absolute SI amplitude, half-power
crossings, bandwidth and Q = peak frequency / bandwidth. The threshold is
peak amplitude / sqrt(2), not half prominence. Separate plots compare spectra,
peak heights and Q for Z and Y; flute playing resonances are admittance peaks,
not impedance maxima. Missing crossings and bandwidths spanning fewer than eight
grid intervals have no reported Q and an explicit status. These are bandwidth
estimates for isolated model resonances, not measured playing stability or
modal-fit Q values. Reduce `SPECTRUM_STEP_HZ` to check numerical convergence.
