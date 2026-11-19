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
shared chimney bevel angle. Each hole's bore radius can decrease from the
preceding radius by up to a configurable fraction (20% by default), but cannot
increase. The decoded bore interpolates between these radii and stays
non-increasing through the terminal taper. When the endhole is fixed, the hole
radii cannot shrink below it. The search also respects the existing ±5% pipe
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
radius can decrease by up to 30% from the last hole's bore radius, and the GA
adds a 20 mm terminal taper when the seed is cylindrical. Bevel remains
unchanged in this example. The [genetic notebook](notebooks/genetic_geometry_optimization.ipynb)
instead enables independent conical holes and Q rewards as described below.

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

Candidate scoring uses CPU worker processes automatically (available logical
CPUs minus one, bounded by population size and the Windows process limit).
Set `workers=1` in `GeneticAlgorithmConfig` for serial execution, or supply a
positive worker count. Worker failures are reported rather than silently
falling back to serial execution.

The best 10 distinct candidates are saved automatically in
`data/ga-winners/` and reused for compatible seed geometry, tuning, notes,
mouthpiece, constraints and scoring settings. Saved candidates are rescored
before use; they do not displace the seed, and a small population admits only
as many as fit. Rescoring the archive can add up to 10 evaluations before the
first generation. Set `winner_directory=None` to disable loading and saving, or
set another directory path. Archives change the initial population, so a fixed
random seed is reproducible only with the same archive snapshot. Restart a
running notebook kernel after updating the optimizer code.

Use [the saved-winner review notebook](notebooks/genetic_winner_results.ipynb)
to select an archive and winner without rerunning the optimization. It
reconstructs the saved seed and mouthpiece, recomputes the leaderboard, checks
body geometry, and compares pitch, half-power Q and spectra on a denser
frequency grid. Both the original and GA-adapted seed are shown. The archive is
read only; passive acoustic predictions still require physical validation.

For the alternative scoring used by the genetic notebook, set
`root_relative_q=True` and `retune_closed_root=False`. Pipe length remains a GA
gene but is not retuned while scoring. Each register's root error is measured
against its target and weighted 10 times; other pitch errors are relative to
that register's actual root. With the default `q_reward_weight=0`, cost includes
pitch errors only and Q is diagnostic. A missing
resonance still incurs a 1200-cent pitch error. `relative_cent_errors` reports
the scoring inputs, while `cent_errors` remains absolute. Despite its legacy
name, `root_relative_q` does not add a Q penalty. Each register uses a
weighted mean squared error: divide its weighted squared-error sum by the
sum of note weights (10 for a root, 1 for each other note). Combine these
means as 60% lower and 40% upper; when only one register is selected, its
share becomes 100%.

The genetic notebook enables `independent_hole_geometry=True` and
`optimize_bevel=True`. Each hole independently varies its chimney height
and 0-15 degree bore-side conical bevel. Configure `max_outer_diameter_m`
(notebook: 0.016 m) and `minimum_wall_m` (0.0009 m). Height is on the
0.1 mm grid between the minimum wall and `outer_diameter/2 - local_bore_radius`.
Local exterior radii may vary; this is not a uniform cylindrical outer surface.
The fixed inlet must fit the body envelope or the optimizer raises an error.
Constraints are reapplied after crossover and mutation.

In this mode, the 5-8 mm hole diameter describes the finger opening.
Following [OpenWInD's conical-hole model](https://files.inria.fr/openwind/docs/examples/technical.Ex8_conical_holes_deviation_pipes.html),
`radius_out = diameter/2` and bore-side `radius = radius_out + height*tan(bevel)`.
The inner opening must fit the local bore, limiting diameters/bevels as needed.
This models a full-height cone, not a short chamfer. Old holes retain their
outward flare; `bevel_side="bore"` explicitly selects inward flaring in
`UndercutTonehole`. Copies, finalization and exports preserve that direction.
With independent geometry disabled, existing seed heights and legacy bevel
behavior remain unchanged.

To reward higher Q, set `q_reward_weight=1.0` and `q_reward_scale=40.0`, as in
the notebook. This requires `root_relative_q=True`. Combined cost is pitch
cost minus the weighted mean of `Q/(Q+scale)`: equal note weights within each
register, then 60% lower / 40% upper (renormalized for a single register).
Higher resolved Q always helps, but the reward cannot exceed its configured
weight. There is no Q threshold or extra low-Q penalty. Unavailable Q earns
zero reward; missing resonances retain their pitch penalty. Negative combined
costs are valid. Results expose `initial_pitch_cost`, `final_pitch_cost`,
`initial_q_reward`, and `final_q_reward` separately from combined cost.
Dense-grid checks and real playing measurements remain necessary.

Scoring revision 4 and explicit genome schemas isolate new winners from old
objectives and conicity conventions. The review notebook supports revisions
1-4; it rescores revisions 1-3 with normalized pitch-only costs and preserves
their original geometry. It uses saved reward/manufacturing settings for new
archives. Historical archives are not rewritten or silently imported into
incompatible searches.

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
