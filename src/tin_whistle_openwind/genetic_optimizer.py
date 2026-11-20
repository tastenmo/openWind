"""Genetic refinement of whistle bore and tonehole geometry."""

from __future__ import annotations

import os
import sys
import json
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

from .geometry import (
    DEFAULT_TEMPERATURE,
    DEFAULT_DATA_DIR,
    MAX_BEVEL_DEG,
    MouthpieceGeometry,
    TaperedBoreGeometry,
    UndercutTonehole,
    WhistleGeometryFactory,
    WhistleInstrument,
)
from .keys import WhistleKeyConfig
from .optimizer import (
    ACTIVE_HOLE_LABELS,
    MAX_HOLE_DIAMETER_M,
    MIN_ADJACENT_HOLE_SPACINGS_M,
    MIN_BOTTOM_END_DISTANCE_M,
    MIN_HOLE_DIAMETER_M,
    ROOT_TUNING_TOLERANCE_CENTS,
    OpenWInDToneholeModel,
    ToneholeLayout,
)
from .resonances import resonance_peak_rows
from .genetic_winners import WinnerArchive
from .tuning import (
    DEGREE_LABELS,
    SECOND_REGISTER_FINGERINGS,
    TuningProfile,
    generate_key_frequencies,
)

MISSING_UPPER_RESONANCE_COST_CENTS = 1200.0
GEOMETRY_RESOLUTION_M = 1e-4
BEVEL_RESOLUTION_DEG = 5.0
ROOT_ERROR_WEIGHT = 10.0


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in genetic run log.")


def _write_run_log(directory: Path, payload: dict[str, Any]) -> Path:
    serialized = json.dumps(payload, default=_json_value, indent=2, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.json"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".ga-run-", suffix=".tmp", delete=False,
        ) as file:
            temporary = file.name
            file.write(serialized)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _normalized_note_weights(
    notes: Sequence[str], root_weight: float,
) -> dict[str, float]:
    if not notes:
        raise ValueError("At least one note is required.")
    register_weights = {"Degree_1": 0.6, "Octave_Degree_1": 0.4}
    roots = {
        note: "Octave_Degree_1" if note in SECOND_REGISTER_FINGERINGS else "Degree_1"
        for note in notes
    }
    note_weights = {
        note: root_weight if note == roots[note] else 1.0
        for note in notes
    }
    totals = {
        root: sum(note_weights[note] for note in notes if roots[note] == root)
        for root in set(roots.values())
    }
    register_total = sum(register_weights[root] for root in totals)
    return {
        note: float(
            register_weights[roots[note]] / register_total
            * note_weights[note] / totals[roots[note]]
        )
        for note in notes
    }


def normalized_pitch_costs(errors: dict[str, float]) -> dict[str, float]:
    """Per-note weighted mean squared costs, with 60/40 registers and 10x roots."""
    weights = _normalized_note_weights(list(errors), ROOT_ERROR_WEIGHT)
    return {note: float(weights[note] * error ** 2) for note, error in errors.items()}


def normalized_q_rewards(
    q_factors: dict[str, float | None], *, weight: float = 1.0, scale: float = 40.0,
) -> dict[str, float]:
    """Bounded higher-Q rewards, with equal notes within 60/40 registers."""
    if not np.isfinite(weight) or weight < 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Q reward weight must be nonnegative and scale must be positive.")
    weights = _normalized_note_weights(list(q_factors), 1.0)
    rewards = {}
    for note, q in q_factors.items():
        if q is not None and (not np.isfinite(q) or q <= 0):
            raise ValueError("Resolved Q must be finite and positive.")
        rewards[note] = 0.0 if q is None else float(
            weight * weights[note] * (1.0 / (1.0 + scale / q))
        )
    return rewards


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Controls for the search; bore-radius fraction caps each downward hole step."""

    population_size: int = 20
    generations: int = 10
    elite_count: int = 2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_scale: float = 0.08
    bore_radius_variation_fraction: float = 0.20
    random_seed: int | None = None
    retune_closed_root: bool = True
    include_second_register: bool = False
    optimize_endhole: bool = False
    optimize_bevel: bool = True
    independent_hole_geometry: bool = False
    max_outer_diameter_m: float = 0.016
    minimum_wall_m: float = 0.0009
    q_reward_weight: float = 0.0
    q_reward_scale: float = 40.0
    root_relative_q: bool = False
    workers: int | None = None
    winner_directory: Path | str | None = DEFAULT_DATA_DIR / "ga-winners"
    logbook_directory: Path | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.population_size, int) or isinstance(self.population_size, bool):
            raise ValueError("population_size must be an integer.")
        if not isinstance(self.generations, int) or isinstance(self.generations, bool):
            raise ValueError("generations must be an integer.")
        if not isinstance(self.elite_count, int) or isinstance(self.elite_count, bool):
            raise ValueError("elite_count must be an integer.")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4.")
        if self.generations < 1:
            raise ValueError("generations must be positive.")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1.")
        for name, value in (
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
            ("mutation_scale", self.mutation_scale),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1].")
        if (
            not np.isfinite(self.bore_radius_variation_fraction)
            or not 0.0 <= self.bore_radius_variation_fraction < 1.0
        ):
            raise ValueError("bore_radius_variation_fraction must be within [0, 1).")
        if self.random_seed is not None and (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer or None.")
        if self.root_relative_q and self.retune_closed_root:
            raise ValueError("root_relative_q requires retune_closed_root=False.")
        for name, value in (
            ("max_outer_diameter_m", self.max_outer_diameter_m),
            ("minimum_wall_m", self.minimum_wall_m),
            ("q_reward_scale", self.q_reward_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.minimum_wall_m >= self.max_outer_diameter_m / 2:
            raise ValueError("minimum_wall_m must be smaller than the outer radius.")
        if not np.isfinite(self.q_reward_weight) or self.q_reward_weight < 0:
            raise ValueError("q_reward_weight must be finite and nonnegative.")
        if self.q_reward_weight and not self.root_relative_q:
            raise ValueError("Q rewards require root_relative_q=True.")
        if self.workers is not None and (
            not isinstance(self.workers, int)
            or isinstance(self.workers, bool)
            or self.workers < 1
        ):
            raise ValueError("workers must be a positive integer or None.")
        if self.winner_directory is not None and (
            not isinstance(self.winner_directory, (str, Path))
            or not str(self.winner_directory).strip()
        ):
            raise ValueError("winner_directory must be a directory path or None.")
        if self.logbook_directory is not None and (
            not isinstance(self.logbook_directory, (str, Path))
            or not str(self.logbook_directory).strip()
        ):
            raise ValueError("logbook_directory must be a directory path or None.")


@dataclass
class GeometryDNA:
    """Legacy 20/21 genes, or 31/32 genes with independent heights and bevels."""

    pipe_length_m: float
    hole_positions_m: np.ndarray
    bore_radii_at_holes_m: np.ndarray
    hole_diameters_m: np.ndarray
    bevel_deg: float
    endhole_radius_m: float | None = None
    chimney_heights_m: np.ndarray | None = None
    hole_bevels_deg: np.ndarray | None = None

    def to_vector(self) -> np.ndarray:
        if (self.chimney_heights_m is None) != (self.hole_bevels_deg is None):
            raise ValueError("Independent chimney heights and bevels must be supplied together.")
        if self.chimney_heights_m is not None:
            if self.hole_bevels_deg is None:
                raise ValueError("Independent hole bevels are required with chimney heights.")
            if self.chimney_heights_m.shape != (6,) or self.hole_bevels_deg.shape != (6,):
                raise ValueError("Independent hole arrays must each contain six values.")
            hole_genes = np.r_[self.chimney_heights_m, self.hole_bevels_deg]
        else:
            hole_genes = np.asarray([self.bevel_deg])
        return np.concatenate(
            (
                np.asarray([self.pipe_length_m]),
                np.asarray(self.hole_positions_m, dtype=float),
                np.asarray(self.bore_radii_at_holes_m, dtype=float),
                np.asarray(self.hole_diameters_m, dtype=float),
                hole_genes,
                np.asarray([] if self.endhole_radius_m is None else [self.endhole_radius_m]),
            )
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "GeometryDNA":
        vector = np.asarray(vector, dtype=float)
        if vector.shape not in ((20,), (21,), (31,), (32,)):
            raise ValueError(
                "A geometry genome must contain 20 genes, or 21 with the endhole, "
                "or 31/32 genes with independent holes."
            )
        independent = len(vector) in (31, 32)
        end_index = 31 if independent else 20
        return cls(
            pipe_length_m=float(vector[0]),
            hole_positions_m=vector[1:7].copy(),
            bore_radii_at_holes_m=vector[7:13].copy(),
            hole_diameters_m=vector[13:19].copy(),
            bevel_deg=float(np.mean(vector[25:31])) if independent else float(vector[19]),
            endhole_radius_m=float(vector[end_index]) if len(vector) == end_index + 1 else None,
            chimney_heights_m=vector[19:25].copy() if independent else None,
            hole_bevels_deg=vector[25:31].copy() if independent else None,
        )

    @property
    def dimensional_values(self) -> np.ndarray:
        values = np.r_[
            self.pipe_length_m, self.hole_positions_m,
            self.bore_radii_at_holes_m, self.hole_diameters_m,
        ]
        if self.chimney_heights_m is not None:
            values = np.r_[values, self.chimney_heights_m]
        if self.endhole_radius_m is not None:
            values = np.r_[values, self.endhole_radius_m]
        return values


@dataclass
class GeneticRepairDiagnostics:
    """Variation and repair rates for offspring proposals in one generation."""

    generation: int
    proposal_count: int
    duplicate_count: int
    gene_names: tuple[str, ...]
    units: tuple[str, ...]
    before_std: np.ndarray
    after_std: np.ndarray
    changed_fraction: np.ndarray


@dataclass
class GeneticOptimizationResult:
    """Best candidate and diagnostics from a genetic geometry search."""

    bore: TaperedBoreGeometry
    layout: ToneholeLayout
    dna: GeometryDNA
    cent_errors: dict[str, float]
    initial_cost: float
    final_cost: float
    initial_fitness: float
    final_fitness: float
    generations: int
    evaluations: int
    root_tuning_satisfied: bool | None
    initial_cent_errors: dict[str, float] = field(default_factory=dict)
    relative_cent_errors: dict[str, float] = field(default_factory=dict)
    initial_relative_cent_errors: dict[str, float] = field(default_factory=dict)
    q_factors: dict[str, float | None] = field(default_factory=dict)
    initial_q_factors: dict[str, float | None] = field(default_factory=dict)
    initial_pitch_cost: float | None = None
    final_pitch_cost: float | None = None
    initial_q_reward: float = 0.0
    final_q_reward: float = 0.0
    repair_diagnostics: list[GeneticRepairDiagnostics] = field(default_factory=list)
    logbook_path: Path | None = None

    @property
    def improved(self) -> bool:
        return self.initial_cost - self.final_cost > 1e-6


@dataclass
class _CandidateScore:
    vector: np.ndarray
    errors: dict[str, float]
    cost: float
    fitness: float
    root_violation: float
    relative_errors: dict[str, float] = field(default_factory=dict)
    q_factors: dict[str, float | None] = field(default_factory=dict)
    pitch_cost: float | None = None
    q_reward: float = 0.0

    @property
    def rank(self) -> tuple[float, float]:
        return self.root_violation, self.cost if self.root_violation else self.fitness


_worker_optimizer: GeneticGeometryOptimizer | None = None


def _initialize_worker(
    key_config: WhistleKeyConfig,
    tuning: TuningProfile,
    bore: TaperedBoreGeometry,
    layout: ToneholeLayout,
    temperature: float,
    mouthpiece: MouthpieceGeometry | None,
    config: GeneticAlgorithmConfig,
) -> None:
    global _worker_optimizer
    _worker_optimizer = GeneticGeometryOptimizer(
        key_config, tuning, bore, layout, temperature, mouthpiece, config
    )


def _score_in_worker(
    task: tuple[np.ndarray, tuple[str, ...]],
) -> _CandidateScore:
    if _worker_optimizer is None:
        raise RuntimeError("Genetic optimization worker is not initialized.")
    vector, notes = task
    return _worker_optimizer._score(
        vector, notes, retune=_worker_optimizer.config.retune_closed_root
    )


class _GeometryGenomeCodec:
    """Encode geometry with 0.1 mm resolution and cascading bore-radius bounds."""

    def __init__(
        self,
        bore: TaperedBoreGeometry,
        layout: ToneholeLayout,
        config: GeneticAlgorithmConfig,
    ) -> None:
        self.seed_bore = bore
        self.seed_layout = layout
        self.config = config
        self._validate_seed_geometry()
        self.seed_layout.validate(self.seed_bore.length_m)

    def _validate_seed_geometry(self) -> None:
        points = np.asarray(self.seed_bore.segments, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.all(np.isfinite(points))
            or points[0, 0] != 0.0
            or np.any(np.diff(points[:, 0]) <= 0.0)
            or np.any(points[:, 1] <= 0.0)
        ):
            raise ValueError(
                "The seed bore must have finite, increasing segments starting at position 0 "
                "with positive radii."
            )
        if len(self.seed_layout.holes) != len(ACTIVE_HOLE_LABELS):
            raise ValueError("A six-hole seed layout is required.")
        if tuple(hole.label for hole in self.seed_layout.holes) != ACTIVE_HOLE_LABELS:
            raise ValueError("Seed holes must be labeled and ordered from L1 to R3.")
        if self.config.independent_hole_geometry:
            maximum_bore = self.config.max_outer_diameter_m / 2 - self.config.minimum_wall_m
            fixed_radii = [points[0, 1]]
            if not self.config.optimize_endhole:
                fixed_radii.append(points[-1, 1])
            if max(fixed_radii) > maximum_bore + 1e-12:
                raise ValueError("Fixed bore exceeds the outer diameter/minimum wall envelope.")

    def _resized_seed(self, length_m: float) -> TaperedBoreGeometry:
        return self.seed_bore.with_length(length_m)

    def _seed_radii_at(
        self, positions_m: np.ndarray, length_m: float
    ) -> np.ndarray:
        resized = self._resized_seed(length_m)
        points = np.asarray(resized.segments, dtype=float)
        return np.interp(positions_m, points[:, 0], points[:, 1])

    def _position_gaps(self) -> np.ndarray:
        return np.asarray(MIN_ADJACENT_HOLE_SPACINGS_M, dtype=float) + 1e-4

    @staticmethod
    def _grid_units(value_m: float, *, ceil: bool = False, floor: bool = False) -> int:
        units = value_m / GEOMETRY_RESOLUTION_M
        if ceil:
            return int(np.ceil(units - 1e-9))
        if floor:
            return int(np.floor(units + 1e-9))
        return int(np.rint(units))

    @classmethod
    def _quantize_bounded(cls, value_m: float, lower_m: float, upper_m: float) -> float:
        lower_units = cls._grid_units(lower_m, ceil=True)
        upper_units = cls._grid_units(upper_m, floor=True)
        if lower_units > upper_units:
            raise ValueError(
                "The requested geometric bounds contain no 0.1 mm resolution value."
            )
        units = int(np.clip(cls._grid_units(value_m), lower_units, upper_units))
        return units * GEOMETRY_RESOLUTION_M

    @staticmethod
    def _quantize_bevel(value_deg: float, maximum_deg: float = MAX_BEVEL_DEG) -> float:
        maximum_units = int(np.floor(maximum_deg / BEVEL_RESOLUTION_DEG + 1e-12))
        units = int(np.clip(
            np.floor(value_deg / BEVEL_RESOLUTION_DEG + 0.5), 0, maximum_units
        ))
        return float(units * BEVEL_RESOLUTION_DEG)

    def _minimum_length_for_bottom_hole(self, position_m: float) -> float:
        terminal_length = (
            self.seed_bore.length_m - self.seed_bore.segments[-2][0]
            if len(self.seed_bore.segments) > 2
            else 0.0
        )
        return position_m + max(MIN_BOTTOM_END_DISTANCE_M, terminal_length) + 1e-4

    def repair(self, vector: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(vector)):
            raise ValueError("Genome genes must all be finite.")
        dna = GeometryDNA.from_vector(vector)
        if (dna.chimney_heights_m is not None) != self.config.independent_hole_geometry:
            raise ValueError("Genome schema does not match independent_hole_geometry.")
        length_m = self._quantize_bounded(
            dna.pipe_length_m,
            0.95 * self.seed_bore.length_m,
            1.05 * self.seed_bore.length_m,
        )

        gap_units = np.ceil(
            self._position_gaps() / GEOMETRY_RESOLUTION_M - 1e-9
        ).astype(int)
        offsets = np.r_[0, np.cumsum(gap_units)]
        first_position_min = self._grid_units(0.018, ceil=True)
        last_position_max = self._grid_units(
            length_m - MIN_BOTTOM_END_DISTANCE_M - 1e-4, floor=True
        )
        resized_seed = self._resized_seed(length_m)
        if len(resized_seed.segments) > 2:
            last_position_max = min(
                last_position_max,
                self._grid_units(resized_seed.segments[-2][0] - 1e-4, floor=True),
            )
        last_transformed_max = last_position_max - offsets[-1]
        if last_transformed_max < first_position_min:
            raise ValueError("Pipe is too short to fit the six-hole spacing constraints.")
        transformed = np.rint(dna.hole_positions_m / GEOMETRY_RESOLUTION_M) - offsets
        transformed = np.clip(transformed, first_position_min, last_transformed_max)
        transformed = np.maximum.accumulate(transformed)
        positions = (transformed + offsets) * GEOMETRY_RESOLUTION_M

        seed_radii = self._seed_radii_at(positions, length_m)
        radius_fraction = self.config.bore_radius_variation_fraction
        radii = np.empty(len(seed_radii), dtype=float)
        previous_radius = (
            self._grid_units(
                min(seed_radii[0], self.seed_bore.segments[0][1]), floor=True
            )
            * GEOMETRY_RESOLUTION_M
        )
        if self.config.independent_hole_geometry:
            previous_radius = min(
                previous_radius,
                self._grid_units(
                    self.config.max_outer_diameter_m / 2 - self.config.minimum_wall_m,
                    floor=True,
                ) * GEOMETRY_RESOLUTION_M,
            )
        fixed_end_radius = (
            0.0 if self.config.optimize_endhole else self.seed_bore.segments[-1][1]
        )
        for index, proposed_radius in enumerate(dna.bore_radii_at_holes_m):
            radii[index] = self._quantize_bounded(
                proposed_radius,
                max(
                    previous_radius * (1.0 - radius_fraction), fixed_end_radius,
                    MIN_HOLE_DIAMETER_M / 2 if self.config.independent_hole_geometry else 0,
                ),
                previous_radius,
            )
            previous_radius = radii[index]
        diameters = np.asarray(
            [
                self._quantize_bounded(
                    diameter, MIN_HOLE_DIAMETER_M,
                    min(MAX_HOLE_DIAMETER_M, 2 * radii[index])
                    if self.config.independent_hole_geometry else MAX_HOLE_DIAMETER_M,
                )
                for index, diameter in enumerate(dna.hole_diameters_m)
            ],
            dtype=float,
        )
        bevel_deg = (
            self._quantize_bevel(dna.bevel_deg)
            if self.config.optimize_bevel else self._seed_bevel()
        )
        heights = bevels = None
        if self.config.independent_hole_geometry:
            if dna.chimney_heights_m is None or dna.hole_bevels_deg is None:
                raise ValueError("Independent hole genomes require six heights and bevels.")
            heights = np.asarray([
                self._quantize_bounded(
                    height, self.config.minimum_wall_m,
                    self.config.max_outer_diameter_m / 2 - radius,
                )
                for height, radius in zip(dna.chimney_heights_m, radii)
            ])
            bevels = np.asarray([
                hole.bevel_deg if not self.config.optimize_bevel else self._quantize_bevel(
                    angle,
                    min(
                        MAX_BEVEL_DEG,
                        np.degrees(
                            np.arctan(max(0.0, radius - diameter / 2) / height)
                        ),
                    ),
                )
                for hole, angle, radius, diameter, height in zip(
                    self.seed_layout.holes, dna.hole_bevels_deg, radii, diameters, heights
                )
            ])
            for index, hole in enumerate(self.seed_layout.holes):
                if not self.config.optimize_bevel and hole.bevel_side == "bore":
                    inner = diameters[index] / 2 + heights[index] * np.tan(np.radians(bevels[index]))
                    if inner > radii[index] + 1e-12:
                        raise ValueError("Fixed bore-side bevel does not fit the local bore.")
        endhole_radius = None
        if self.config.optimize_endhole:
            gene = (
                self.seed_bore.segments[-1][1]
                if dna.endhole_radius_m is None
                else dna.endhole_radius_m
            )
            endhole_radius = self._quantize_bounded(
                gene, radii[-1] * 0.70, radii[-1]
            )
        repaired = GeometryDNA(
            length_m, positions, radii, diameters, bevel_deg, endhole_radius,
            chimney_heights_m=heights, hole_bevels_deg=bevels,
        ).to_vector()
        return repaired

    def _seed_bevel(self) -> float:
        return float(np.mean([hole.bevel_deg for hole in self.seed_layout.holes]))

    def from_geometry(self) -> np.ndarray:
        positions = self.seed_layout.positions_m
        length_m = self.seed_bore.length_m
        radii = self._seed_radii_at(positions, length_m)
        diameters = self.seed_layout.diameters_m
        bevel = self._seed_bevel()
        return self.repair(
            GeometryDNA(
                length_m, positions, radii, diameters, bevel,
                chimney_heights_m=np.asarray([
                    hole.chimney_height_m for hole in self.seed_layout.holes
                ]) if self.config.independent_hole_geometry else None,
                hole_bevels_deg=np.asarray([
                    hole.bevel_deg for hole in self.seed_layout.holes
                ]) if self.config.independent_hole_geometry else None,
            ).to_vector()
        )

    def decode(self, vector: np.ndarray) -> tuple[TaperedBoreGeometry, ToneholeLayout]:
        dna = GeometryDNA.from_vector(self.repair(vector))
        seed_profile = self._resized_seed(dna.pipe_length_m)
        seed_points = np.asarray(seed_profile.segments, dtype=float)
        knot_positions = np.unique(
            np.concatenate((seed_points[:, 0], dna.hole_positions_m))
        )
        end_radius = (
            dna.endhole_radius_m
            if dna.endhole_radius_m is not None
            else seed_points[-1, 1]
        )
        profile_positions = np.r_[0.0, dna.hole_positions_m, dna.pipe_length_m]
        profile_radii = np.r_[seed_points[0, 1], dna.bore_radii_at_holes_m, end_radius]
        if len(seed_points) > 2 or dna.endhole_radius_m is not None:
            terminal_taper_start = (
                seed_points[-2, 0] if len(seed_points) > 2 else dna.pipe_length_m - 0.020
            )
            knot_positions = np.union1d(knot_positions, [terminal_taper_start])
            profile_positions = np.r_[
                0.0,
                dna.hole_positions_m,
                terminal_taper_start,
                dna.pipe_length_m,
            ]
            profile_radii = np.r_[
                seed_points[0, 1], dna.bore_radii_at_holes_m,
                dna.bore_radii_at_holes_m[-1], end_radius,
            ]
        knot_radii = np.interp(knot_positions, profile_positions, profile_radii)
        bore = TaperedBoreGeometry(
            name=self.seed_bore.name,
            segments=[
                (float(position), float(radius))
                for position, radius in zip(knot_positions, knot_radii)
            ],
        )
        holes = ToneholeLayout(
            [
                UndercutTonehole(
                    label=hole.label,
                    position_m=float(dna.hole_positions_m[index]),
                    diameter_m=float(dna.hole_diameters_m[index]),
                    chimney_height_m=(
                        float(dna.chimney_heights_m[index])
                        if dna.chimney_heights_m is not None else hole.chimney_height_m
                    ),
                    bevel_deg=(
                        float(dna.hole_bevels_deg[index]) if dna.hole_bevels_deg is not None
                        else dna.bevel_deg if self.config.optimize_bevel else hole.bevel_deg
                    ),
                    bevel_side=(
                        "bore" if self.config.independent_hole_geometry and self.config.optimize_bevel
                        else hole.bevel_side
                    ),
                )
                for index, hole in enumerate(self.seed_layout.holes)
            ]
        )
        holes.validate(dna.pipe_length_m)
        return bore, holes


class GeneticGeometryOptimizer:
    """Search valid bore and tonehole designs with a continuous genetic algorithm."""

    def __init__(
        self,
        key_config: WhistleKeyConfig,
        tuning: TuningProfile,
        bore: TaperedBoreGeometry,
        layout: ToneholeLayout,
        temperature: float = DEFAULT_TEMPERATURE,
        mouthpiece: MouthpieceGeometry | None = None,
        config: GeneticAlgorithmConfig | None = None,
    ) -> None:
        self.key_config = key_config
        self.tuning = tuning
        self.temperature = temperature
        self.mouthpiece = mouthpiece
        self.config = config or GeneticAlgorithmConfig()
        self.codec = _GeometryGenomeCodec(bore, layout, self.config)
        self._evaluations = 0
        self._cache: dict[tuple[float, ...], _CandidateScore] = {}
        self._repair_samples: list[tuple[np.ndarray, np.ndarray]] | None = None

    def _cent_errors(
        self, vector: np.ndarray, notes: Sequence[str]
    ) -> dict[str, float]:
        bore, layout = self.codec.decode(vector)
        acoustic_bore: TaperedBoreGeometry | WhistleInstrument = bore
        if self.mouthpiece is not None:
            acoustic_bore = WhistleGeometryFactory(
                temperature=self.temperature
            ).create_tapered_instrument(self.mouthpiece, bore)
        model = OpenWInDToneholeModel(
            acoustic_bore,
            layout,
            self.temperature,
            hole_position_offset_m=self.mouthpiece.l_mouth if self.mouthpiece else 0.0,
        )
        base_targets = generate_key_frequencies(self.key_config, "12-TET")
        errors: dict[str, float] = {}
        for note in notes:
            fingering = SECOND_REGISTER_FINGERINGS.get(note, note)
            if note in SECOND_REGISTER_FINGERINGS and note != "Octave_Degree_1":
                target = 2.0 * self.tuning.get_target_frequency(fingering, base_targets[fingering])
            else:
                target = self.tuning.get_target_frequency(note, base_targets[note])
            upper = note in SECOND_REGISTER_FINGERINGS and self.config.include_second_register
            response = model.response(
                fingering,
                target,
                f_min=(0.8 if upper else 0.75) * target,
                f_max=(1.2 if upper else 1.25) * target,
                n_points=300,
            )
            if upper and not len(find_peaks(response.admittance)[0]):
                errors[note] = MISSING_UPPER_RESONANCE_COST_CENTS
                continue
            errors[note] = float(
                1200.0 * np.log2(response.peak_frequency_hz / target)
            )
        return errors

    def _relative_q_score(
        self, vector: np.ndarray, notes: Sequence[str]
    ) -> _CandidateScore:
        bore, layout = self.codec.decode(vector)
        acoustic_bore: TaperedBoreGeometry | WhistleInstrument = bore
        if self.mouthpiece is not None:
            acoustic_bore = WhistleGeometryFactory(
                temperature=self.temperature
            ).create_tapered_instrument(self.mouthpiece, bore)
        model = OpenWInDToneholeModel(
            acoustic_bore,
            layout,
            self.temperature,
            hole_position_offset_m=self.mouthpiece.l_mouth if self.mouthpiece else 0.0,
        )
        base_targets = generate_key_frequencies(self.key_config, "12-TET")
        requested = list(dict.fromkeys(notes))
        for note in notes:
            root = "Octave_Degree_1" if note in SECOND_REGISTER_FINGERINGS else "Degree_1"
            if root not in requested:
                requested.append(root)

        errors: dict[str, float] = {}
        q_factors: dict[str, float | None] = {}
        missing: set[str] = set()
        for note in requested:
            fingering = SECOND_REGISTER_FINGERINGS.get(note, note)
            if note in SECOND_REGISTER_FINGERINGS and note != "Octave_Degree_1":
                target = 2.0 * self.tuning.get_target_frequency(
                    fingering, base_targets[fingering]
                )
            else:
                target = self.tuning.get_target_frequency(note, base_targets[note])
            upper = note in SECOND_REGISTER_FINGERINGS
            response = model.response(
                fingering,
                target,
                f_min=(0.8 if upper else 0.75) * target,
                f_max=(1.2 if upper else 1.25) * target,
                n_points=1201,
            )
            if not len(find_peaks(response.admittance)[0]):
                errors[note] = MISSING_UPPER_RESONANCE_COST_CENTS
                q_factors[note] = None
                missing.add(note)
                continue
            errors[note] = float(
                1200.0 * np.log2(response.peak_frequency_hz / target)
            )
            rows = resonance_peak_rows(response.frequencies_hz, response.impedance, "Y")
            peak = min(rows, key=lambda row: abs(float(row["Peak (Hz)"]) - target))
            q = peak["Q"]
            q_factors[note] = float(q) if q is not None else None

        relative_errors: dict[str, float] = {}
        for note in notes:
            root = "Octave_Degree_1" if note in SECOND_REGISTER_FINGERINGS else "Degree_1"
            if note == root:
                error = errors[note]
            elif note in missing or root in missing:
                error = MISSING_UPPER_RESONANCE_COST_CENTS
            else:
                error = errors[note] - errors[root]
            relative_errors[note] = error
        pitch_cost = float(sum(normalized_pitch_costs(relative_errors).values()))
        q_reward = float(sum(normalized_q_rewards(
            {note: q_factors[note] for note in notes},
            weight=self.config.q_reward_weight, scale=self.config.q_reward_scale,
        ).values()))
        cost = pitch_cost - q_reward
        if not np.isfinite(cost):
            raise ValueError("Non-finite tuning cost during genetic optimization.")
        return _CandidateScore(
            vector, {note: errors[note] for note in notes}, cost, cost, 0.0,
            relative_errors, {note: q_factors[note] for note in notes},
            pitch_cost, q_reward,
        )

    def _retune_length(self, vector: np.ndarray) -> tuple[np.ndarray, float]:
        dna = GeometryDNA.from_vector(vector)
        length = dna.pipe_length_m
        seed_length = self.codec.seed_bore.length_m
        lower = min(length, max(
            0.95 * seed_length,
            self.codec._minimum_length_for_bottom_hole(dna.hole_positions_m[-1]),
        ))
        upper = 1.05 * seed_length
        radius_ratios = dna.bore_radii_at_holes_m / self.codec._seed_radii_at(
            dna.hole_positions_m, length
        )
        trials: dict[float, tuple[np.ndarray, float]] = {}

        def root_error(trial_length: float) -> float:
            if trial_length not in trials:
                trial = vector.copy()
                trial[0] = trial_length
                trial[7:13] = radius_ratios * self.codec._seed_radii_at(
                    dna.hole_positions_m, trial_length
                )
                trial = self.codec.repair(trial)
                error = self._cent_errors(trial, ("Degree_1",))["Degree_1"]
                if not np.isfinite(error):
                    raise ValueError("Non-finite closed-root error during length retuning.")
                trials[trial_length] = (trial, error)
            return trials[trial_length][1]

        start_error = root_error(length)
        if abs(start_error) <= 1e-3:
            return trials[length]

        endpoints = (upper, lower) if start_error > 0 else (lower, upper)
        for endpoint in endpoints:
            error = root_error(endpoint)
            if abs(error) <= 1e-3:
                return trials[endpoint]
            if np.signbit(start_error) != np.signbit(error):
                left, right = sorted((length, endpoint))
                tuned_length = brentq(root_error, left, right, xtol=1e-8)
                if abs(root_error(tuned_length)) <= ROOT_TUNING_TOLERANCE_CENTS:
                    return trials[tuned_length]
        # If the target cannot be bracketed, keep the closest observed trial.
        return min(trials.values(), key=lambda trial: abs(trial[1]))

    def _score(
        self, vector: np.ndarray, notes: Sequence[str], *, retune: bool
    ) -> _CandidateScore:
        if self.config.root_relative_q:
            return self._relative_q_score(vector, notes)
        root_selected = "Degree_1" in notes
        if retune and root_selected:
            vector, root_error = self._retune_length(vector)
            other_notes = tuple(note for note in notes if note != "Degree_1")
            errors = self._cent_errors(vector, other_notes) if other_notes else {}
            errors["Degree_1"] = root_error
            errors = {note: errors[note] for note in notes}
        else:
            errors = self._cent_errors(vector, notes)
        cost = float(sum(error * error for error in errors.values()))
        if not np.isfinite(cost):
            raise ValueError("Non-finite tuning cost during genetic optimization.")
        violation = (
            max(0.0, abs(errors["Degree_1"]) - ROOT_TUNING_TOLERANCE_CENTS)
            if root_selected else 0.0
        )
        if self.config.retune_closed_root:
            fitness = float("inf") if violation else cost
            return _CandidateScore(vector, errors, cost, fitness, violation)
        fitness = cost + 1e6 * violation ** 2
        return _CandidateScore(vector, errors, cost, fitness, 0.0)

    def _evaluate(self, vector: np.ndarray, notes: Sequence[str]) -> _CandidateScore:
        repaired = self.codec.repair(vector)
        key = tuple(float(value) for value in repaired)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._score(
            repaired, notes, retune=self.config.retune_closed_root
        )
        self._cache[key] = result
        self._evaluations += 1
        return result

    def _evaluate_population(
        self,
        population: Sequence[np.ndarray],
        notes: tuple[str, ...],
        executor: ProcessPoolExecutor | None,
    ) -> list[_CandidateScore]:
        if executor is None:
            return [self._evaluate(candidate, notes) for candidate in population]
        keys = []
        uncached: dict[tuple[float, ...], np.ndarray] = {}
        for candidate in population:
            repaired = self.codec.repair(candidate)
            key = self._individual_key(repaired)
            keys.append(key)
            if key not in self._cache:
                uncached[key] = repaired
        results = list(executor.map(
            _score_in_worker, ((candidate, notes) for candidate in uncached.values())
        ))
        for key, result in zip(uncached, results):
            self._cache[key] = result
            self._evaluations += 1
        return [self._cache[key] for key in keys]

    def _worker_count(self) -> int:
        if self.config.workers is None:
            available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
            workers = max(1, available - 1)
        else:
            workers = self.config.workers
        limit = 61 if sys.platform == "win32" else self.config.population_size
        return min(workers, self.config.population_size, limit)

    def _mutation_scales(self, vector: np.ndarray) -> np.ndarray:
        dna = GeometryDNA.from_vector(vector)
        length_scale = self.config.mutation_scale
        seed_radii = self.codec._seed_radii_at(
            dna.hole_positions_m, dna.pipe_length_m
        )
        radius_range = (
            2.0
            * seed_radii
            * self.config.bore_radius_variation_fraction
        )
        scales = np.r_[
            dna.pipe_length_m * 0.01 * length_scale,
            np.full(6, dna.pipe_length_m * 0.015 * length_scale),
            radius_range * length_scale,
            np.full(
                6,
                (MAX_HOLE_DIAMETER_M - MIN_HOLE_DIAMETER_M) * length_scale,
            ),
        ]
        if dna.chimney_heights_m is not None:
            spans = np.maximum(
                0.0,
                self.config.max_outer_diameter_m / 2
                - dna.bore_radii_at_holes_m - self.config.minimum_wall_m,
            )
            scales = np.r_[
                scales, spans * length_scale,
                np.full(
                    6,
                    MAX_BEVEL_DEG * length_scale if self.config.optimize_bevel else 0.0,
                ),
            ]
        else:
            scales = np.r_[
                scales,
                MAX_BEVEL_DEG * length_scale if self.config.optimize_bevel else 0.0,
            ]
        if dna.endhole_radius_m is not None:
            span = 2.0 * self.codec.seed_bore.segments[-1][1] * self.config.bore_radius_variation_fraction
            scales = np.r_[scales, span * length_scale]
        return scales

    def _initial_population(
        self,
        seed: np.ndarray,
        rng: np.random.Generator,
        winners: Sequence[np.ndarray] = (),
    ) -> np.ndarray:
        population = [seed.copy()]
        seen = {self._individual_key(seed)}
        for winner in winners:
            candidate = self.codec.repair(winner)
            key = self._individual_key(candidate)
            if key not in seen and len(population) < self.config.population_size:
                population.append(candidate)
                seen.add(key)
        while len(population) < self.config.population_size:
            population.append(self._new_random_individual(seed, rng, seen))
        return np.asarray(population)

    def _archive_context(self, notes: tuple[str, ...]) -> dict[str, Any]:
        return {
            "scoring_revision": 4,
            "genome_schema": 2 if self.config.independent_hole_geometry else 1,
            "key": asdict(self.key_config),
            "tuning": asdict(self.tuning),
            "bore": asdict(self.codec.seed_bore),
            "holes": [asdict(hole) for hole in self.codec.seed_layout.holes],
            "temperature": self.temperature,
            "mouthpiece": asdict(self.mouthpiece) if self.mouthpiece is not None else None,
            "notes": notes,
            "geometry_and_scoring": {
                "bore_radius_variation_fraction": self.config.bore_radius_variation_fraction,
                "optimize_endhole": self.config.optimize_endhole,
                "optimize_bevel": self.config.optimize_bevel,
                "max_bevel_deg": MAX_BEVEL_DEG,
                "bevel_resolution_deg": BEVEL_RESOLUTION_DEG,
                "retune_closed_root": self.config.retune_closed_root,
                "include_second_register": self.config.include_second_register,
                "root_relative_q": self.config.root_relative_q,
                "independent_hole_geometry": self.config.independent_hole_geometry,
                "max_outer_diameter_m": self.config.max_outer_diameter_m,
                "minimum_wall_m": self.config.minimum_wall_m,
                "q_reward_weight": self.config.q_reward_weight,
                "q_reward_scale": self.config.q_reward_scale,
            },
        }
    def _winner_archive(self, notes: tuple[str, ...]) -> WinnerArchive | None:
        if self.config.winner_directory is None:
            return None
        return WinnerArchive(Path(self.config.winner_directory), self._archive_context(notes))

    @staticmethod
    def _individual_key(vector: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in vector)

    def _new_random_individual(
        self,
        seed: np.ndarray,
        rng: np.random.Generator,
        seen: set[tuple[float, ...]],
    ) -> np.ndarray:
        scales = np.maximum(
            self._mutation_scales(seed) * 4.0,
            2.0 * GEOMETRY_RESOLUTION_M,
        )
        for _ in range(1000):
            candidate = self.codec.repair(
                seed + rng.normal(size=seed.shape) * scales
            )
            key = self._individual_key(candidate)
            if key not in seen:
                seen.add(key)
                return candidate
        raise RuntimeError(
            "Unable to generate a unique individual at 0.1 mm resolution."
        )

    @staticmethod
    def _tournament(
        population: np.ndarray,
        scores: Sequence[_CandidateScore],
        rng: np.random.Generator,
        tournament_size: int = 3,
    ) -> np.ndarray:
        contestants = rng.integers(0, len(population), size=tournament_size)
        winner = min(contestants, key=lambda index: scores[index].rank)
        return population[winner]

    def _child(
        self,
        first: np.ndarray,
        second: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        child = first.copy()
        if rng.random() < self.config.crossover_rate:
            alpha = rng.random(size=first.shape)
            child = alpha * first + (1.0 - alpha) * second
        mutation_mask = rng.random(size=child.shape) < self.config.mutation_rate
        scales = self._mutation_scales(child)
        child[mutation_mask] += (
            rng.normal(size=np.count_nonzero(mutation_mask))
            * scales[mutation_mask]
        )
        repaired = self.codec.repair(child)
        if self._repair_samples is not None:
            self._repair_samples.append((child.copy(), repaired.copy()))
        return repaired

    def _summarize_repair(
        self, generation: int, seed: np.ndarray, duplicate_count: int,
        samples: list[tuple[np.ndarray, np.ndarray]],
    ) -> GeneticRepairDiagnostics:
        names = (
            ["pipe_length"]
            + [f"position_{label}" for label in ACTIVE_HOLE_LABELS]
            + [f"bore_radius_{label}" for label in ACTIVE_HOLE_LABELS]
            + [f"diameter_{label}" for label in ACTIVE_HOLE_LABELS]
        )
        units = ["mm"] * len(names)
        dna = GeometryDNA.from_vector(seed)
        if dna.chimney_heights_m is not None:
            names += [f"chimney_{label}" for label in ACTIVE_HOLE_LABELS]
            names += [f"bevel_{label}" for label in ACTIVE_HOLE_LABELS]
            units += ["mm"] * 6 + ["deg"] * 6
        else:
            names.append("bevel")
            units.append("deg")
        if dna.endhole_radius_m is not None:
            names.append("endhole_radius")
            units.append("mm")
        before = np.asarray([sample[0] for sample in samples])
        after = np.asarray([sample[1] for sample in samples])
        scale = np.asarray([1000.0 if unit == "mm" else 1.0 for unit in units])
        return GeneticRepairDiagnostics(
            generation=generation,
            proposal_count=len(samples),
            duplicate_count=duplicate_count,
            gene_names=tuple(names),
            units=tuple(units),
            before_std=np.std(before, axis=0) * scale,
            after_std=np.std(after, axis=0) * scale,
            changed_fraction=np.mean(np.abs(before - after) > 1e-12, axis=0),
        )

    def optimize(
        self,
        notes: Sequence[str] | None = None,
        progress_callback: Callable[[int, float, float, int], None] | None = None,
    ) -> GeneticOptimizationResult:
        """Run the search and optionally report each generation's best score.

        The callback receives the 1-based generation, best cost, best fitness,
        and number of unique candidates scored so far.
        """
        available_notes = list(DEGREE_LABELS)
        if self.config.include_second_register:
            available_notes.extend(note for note in SECOND_REGISTER_FINGERINGS if note not in DEGREE_LABELS)
        selected_notes = tuple(available_notes if notes is None else notes)
        if not selected_notes:
            raise ValueError("At least one tuning note is required.")
        unknown_notes = set(selected_notes) - set(available_notes)
        if unknown_notes:
            raise ValueError(f"Unknown tuning notes: {sorted(unknown_notes)}.")

        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        self._cache.clear()
        self._evaluations = 0
        repair_diagnostics: list[GeneticRepairDiagnostics] = []
        generation_history: list[dict[str, int | float]] = []
        rng = np.random.default_rng(self.config.random_seed)
        initial_rng_state = rng.bit_generator.state
        seed = self.codec.from_geometry()
        baseline = self._score(seed, selected_notes, retune=False)
        archive = self._winner_archive(selected_notes)
        saved = archive.load() if archive is not None else []
        if archive is not None and any(len(genome) != len(seed) for genome in saved):
            raise ValueError(f"Invalid genome length in genetic winner archive: {archive.path}")

        with ExitStack() as stack:
            executor = None
            workers = self._worker_count()
            if workers > 1:
                executor = stack.enter_context(ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_worker,
                    initargs=(
                        self.key_config, self.tuning, self.codec.seed_bore,
                        self.codec.seed_layout, self.temperature, self.mouthpiece,
                        self.config,
                    ),
                ))
            if saved:
                scores = self._evaluate_population(
                    [np.asarray(genome, dtype=float) for genome in saved],
                    selected_notes,
                    executor,
                )
                winners = [score.vector for score in sorted(scores, key=lambda score: score.rank)]
                population = self._initial_population(seed, rng, winners)
            else:
                population = self._initial_population(seed, rng)
            try:
                best = self._evolve(
                    population, seed, selected_notes, rng, executor, progress_callback,
                    repair_diagnostics, generation_history,
                )
            finally:
                self._repair_samples = None

        if baseline.rank <= best.rank:
            best = baseline
        if archive is not None:
            archive.merge([
                (self._individual_key(score.vector), score.rank)
                for score in [baseline, *self._cache.values()]
            ])
        best_vector = best.vector
        bore, layout = self.codec.decode(best_vector)
        root_error = best.errors.get("Degree_1")
        root_satisfied = (
            None
            if root_error is None
            else bool(abs(root_error) <= ROOT_TUNING_TOLERANCE_CENTS)
        )
        result = GeneticOptimizationResult(
            bore=bore,
            layout=layout,
            dna=GeometryDNA.from_vector(best_vector),
            cent_errors=best.errors,
            initial_cost=baseline.cost,
            final_cost=best.cost,
            initial_fitness=baseline.fitness,
            final_fitness=best.fitness,
            generations=self.config.generations,
            evaluations=self._evaluations,
            root_tuning_satisfied=root_satisfied,
            initial_cent_errors=baseline.errors,
            relative_cent_errors=best.relative_errors,
            initial_relative_cent_errors=baseline.relative_errors,
            q_factors=best.q_factors,
            initial_q_factors=baseline.q_factors,
            initial_pitch_cost=baseline.pitch_cost,
            final_pitch_cost=best.pitch_cost,
            initial_q_reward=baseline.q_reward,
            final_q_reward=best.q_reward,
            repair_diagnostics=repair_diagnostics,
        )
        if self.config.logbook_directory is not None:
            result_data = asdict(result)
            del result_data["logbook_path"]
            result.logbook_path = _write_run_log(
                Path(self.config.logbook_directory),
                {
                    "schema_version": 1,
                    "started_at_utc": started_at.isoformat(),
                    "elapsed_seconds": perf_counter() - started,
                    "config": asdict(self.config),
                    "archive": {
                        "path": archive.path if archive is not None else None,
                        "context": self._archive_context(selected_notes),
                        "saved_genomes_at_start": saved,
                    },
                    "initial_rng_state": initial_rng_state,
                    "seed_genome": seed,
                    "generation_history": generation_history,
                    "result": result_data,
                },
            )
        return result

    def _evolve(
        self,
        population: np.ndarray,
        seed: np.ndarray,
        selected_notes: tuple[str, ...],
        rng: np.random.Generator,
        executor: ProcessPoolExecutor | None,
        progress_callback: Callable[[int, float, float, int], None] | None,
        repair_diagnostics: list[GeneticRepairDiagnostics],
        generation_history: list[dict[str, int | float]],
    ) -> _CandidateScore:
        for generation in range(1, self.config.generations + 1):
            scores = self._evaluate_population(population, selected_notes, executor)
            generation_best = min(scores, key=lambda score: score.rank)
            generation_history.append({
                "generation": generation,
                "best_cost": generation_best.cost,
                "best_fitness": generation_best.fitness,
                "unique_candidates_scored": self._evaluations,
            })
            if progress_callback is not None:
                progress_callback(
                    generation,
                    generation_best.cost,
                    generation_best.fitness,
                    self._evaluations,
                )
            elite_indices = sorted(
                range(len(population)), key=lambda index: scores[index].rank
            )[: self.config.elite_count]
            next_population = [population[index].copy() for index in elite_indices]
            seen = {self._individual_key(candidate) for candidate in next_population}
            samples: list[tuple[np.ndarray, np.ndarray]] = []
            self._repair_samples = samples
            duplicate_count = 0
            while len(next_population) < self.config.population_size:
                first = self._tournament(population, scores, rng)
                second = self._tournament(population, scores, rng)
                child = self._child(first, second, rng)
                key = self._individual_key(child)
                if key in seen:
                    duplicate_count += 1
                    child = self._new_random_individual(seed, rng, seen)
                else:
                    seen.add(key)
                next_population.append(child)
            self._repair_samples = None
            if samples:
                repair_diagnostics.append(
                    self._summarize_repair(generation, seed, duplicate_count, samples)
                )
            population = np.asarray(next_population)

        return min(
            self._evaluate_population(population, selected_notes, executor),
            key=lambda score: score.rank,
        )


def optimize_geometry_genetically(
    key_config: WhistleKeyConfig,
    bore: TaperedBoreGeometry,
    layout: ToneholeLayout,
    tuning: TuningProfile,
    temperature: float = DEFAULT_TEMPERATURE,
    mouthpiece: MouthpieceGeometry | None = None,
    config: GeneticAlgorithmConfig | None = None,
    notes: Sequence[str] | None = None,
    progress_callback: Callable[[int, float, float, int], None] | None = None,
) -> GeneticOptimizationResult:
    """Optimize a geometry seed using a NumPy-based genetic search."""
    return GeneticGeometryOptimizer(
        key_config=key_config,
        tuning=tuning,
        bore=bore,
        layout=layout,
        temperature=temperature,
        mouthpiece=mouthpiece,
        config=config,
    ).optimize(notes, progress_callback=progress_callback)
