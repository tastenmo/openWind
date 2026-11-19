"""Genetic refinement of whistle bore and tonehole geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

from .geometry import (
    DEFAULT_TEMPERATURE,
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
from .tuning import (
    DEGREE_LABELS,
    SECOND_REGISTER_FINGERINGS,
    TuningProfile,
    generate_key_frequencies,
)

MISSING_UPPER_RESONANCE_COST_CENTS = 1200.0


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Controls for the continuous, real-valued genetic search."""

    population_size: int = 20
    generations: int = 10
    elite_count: int = 2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_scale: float = 0.08
    bore_radius_variation_fraction: float = 0.05
    random_seed: int | None = None
    retune_closed_root: bool = True
    include_second_register: bool = False
    optimize_endhole: bool = False
    optimize_bevel: bool = True

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


@dataclass
class GeometryDNA:
    """Real-valued genes for pipe length, six holes, their bore radii, and bevel."""

    pipe_length_m: float
    hole_positions_m: np.ndarray
    bore_radii_at_holes_m: np.ndarray
    hole_diameters_m: np.ndarray
    bevel_deg: float
    endhole_radius_m: float | None = None

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray([self.pipe_length_m]),
                np.asarray(self.hole_positions_m, dtype=float),
                np.asarray(self.bore_radii_at_holes_m, dtype=float),
                np.asarray(self.hole_diameters_m, dtype=float),
                np.asarray([self.bevel_deg]),
                np.asarray([] if self.endhole_radius_m is None else [self.endhole_radius_m]),
            )
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "GeometryDNA":
        vector = np.asarray(vector, dtype=float)
        if vector.shape not in ((20,), (21,)):
            raise ValueError("A geometry genome must contain 20 genes, or 21 with the endhole.")
        return cls(
            pipe_length_m=float(vector[0]),
            hole_positions_m=vector[1:7].copy(),
            bore_radii_at_holes_m=vector[7:13].copy(),
            hole_diameters_m=vector[13:19].copy(),
            bevel_deg=float(vector[19]),
            endhole_radius_m=float(vector[20]) if len(vector) == 21 else None,
        )


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

    @property
    def rank(self) -> tuple[float, float]:
        return self.root_violation, self.cost if self.root_violation else self.fitness


class _GeometryGenomeCodec:
    """Encode and decode valid geometry while constraining candidate variation."""

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
        length_m = float(
            np.clip(
                dna.pipe_length_m,
                0.95 * self.seed_bore.length_m,
                1.05 * self.seed_bore.length_m,
            )
        )

        gaps = self._position_gaps()
        offsets = np.r_[0.0, np.cumsum(gaps)]
        first_position_min = 0.018
        last_position_max = length_m - MIN_BOTTOM_END_DISTANCE_M - 1e-4
        resized_seed = self._resized_seed(length_m)
        if len(resized_seed.segments) > 2:
            last_position_max = min(
                last_position_max,
                resized_seed.segments[-2][0] - 1e-4,
            )
        last_transformed_max = last_position_max - offsets[-1]
        if last_transformed_max < first_position_min:
            raise ValueError("Pipe is too short to fit the six-hole spacing constraints.")
        transformed = np.asarray(dna.hole_positions_m, dtype=float) - offsets
        transformed = np.clip(
            transformed, first_position_min, last_transformed_max
        )
        transformed = np.maximum.accumulate(transformed)
        positions = transformed + offsets

        seed_radii = self._seed_radii_at(positions, length_m)
        radius_fraction = self.config.bore_radius_variation_fraction
        radii = np.clip(
            dna.bore_radii_at_holes_m,
            seed_radii * (1.0 - radius_fraction),
            seed_radii * (1.0 + radius_fraction),
        )
        diameters = np.clip(
            dna.hole_diameters_m, MIN_HOLE_DIAMETER_M, MAX_HOLE_DIAMETER_M
        )
        bevel_deg = float(np.clip(dna.bevel_deg, 0.0, 15.0))
        if not self.config.optimize_bevel:
            bevel_deg = self._seed_bevel()
        endhole_radius = None
        if self.config.optimize_endhole:
            seed_radius = self.seed_bore.segments[-1][1]
            gene = seed_radius if dna.endhole_radius_m is None else dna.endhole_radius_m
            endhole_radius = float(np.clip(
                gene, seed_radius * (1.0 - radius_fraction),
                seed_radius * (1.0 + radius_fraction),
            ))
        repaired = GeometryDNA(
            length_m, positions, radii, diameters, bevel_deg, endhole_radius
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
            GeometryDNA(length_m, positions, radii, diameters, bevel).to_vector()
        )

    def decode(self, vector: np.ndarray) -> tuple[TaperedBoreGeometry, ToneholeLayout]:
        dna = GeometryDNA.from_vector(self.repair(vector))
        seed_profile = self._resized_seed(dna.pipe_length_m)
        seed_points = np.asarray(seed_profile.segments, dtype=float)
        seed_radii = np.interp(dna.hole_positions_m, seed_points[:, 0], seed_points[:, 1])
        radius_scales = dna.bore_radii_at_holes_m / seed_radii
        knot_positions = np.unique(
            np.concatenate((seed_points[:, 0], dna.hole_positions_m))
        )
        base_radii = np.interp(knot_positions, seed_points[:, 0], seed_points[:, 1])
        scale_positions = np.r_[0.0, dna.hole_positions_m, dna.pipe_length_m]
        scale_values = np.r_[1.0, radius_scales, 1.0]
        if len(seed_points) > 2 or dna.endhole_radius_m is not None:
            terminal_taper_start = (
                seed_points[-2, 0] if len(seed_points) > 2 else dna.pipe_length_m - 0.020
            )
            knot_positions = np.union1d(knot_positions, [terminal_taper_start])
            base_radii = np.interp(knot_positions, seed_points[:, 0], seed_points[:, 1])
            scale_positions = np.r_[
                0.0,
                dna.hole_positions_m,
                terminal_taper_start,
                dna.pipe_length_m,
            ]
            scale_values = np.r_[1.0, radius_scales, 1.0, 1.0]
        knot_scales = np.interp(knot_positions, scale_positions, scale_values)
        knot_radii = base_radii * knot_scales
        if dna.endhole_radius_m is not None:
            knot_radii[-1] = dna.endhole_radius_m
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
                    chimney_height_m=hole.chimney_height_m,
                    bevel_deg=dna.bevel_deg if self.config.optimize_bevel else hole.bevel_deg,
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
            15.0 * length_scale if self.config.optimize_bevel else 0.0,
        ]
        if dna.endhole_radius_m is not None:
            span = 2.0 * self.codec.seed_bore.segments[-1][1] * self.config.bore_radius_variation_fraction
            scales = np.r_[scales, span * length_scale]
        return scales

    def _initial_population(
        self, seed: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        population = np.repeat(seed[None, :], self.config.population_size, axis=0)
        scales = self._mutation_scales(seed) * 4.0
        for index in range(1, self.config.population_size):
            population[index] = self.codec.repair(
                seed + rng.normal(size=seed.shape) * scales
            )
        return population

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
        return self.codec.repair(child)

    def optimize(
        self, notes: Sequence[str] | None = None
    ) -> GeneticOptimizationResult:
        available_notes = list(DEGREE_LABELS)
        if self.config.include_second_register:
            available_notes.extend(note for note in SECOND_REGISTER_FINGERINGS if note not in DEGREE_LABELS)
        selected_notes = tuple(available_notes if notes is None else notes)
        if not selected_notes:
            raise ValueError("At least one tuning note is required.")
        unknown_notes = set(selected_notes) - set(available_notes)
        if unknown_notes:
            raise ValueError(f"Unknown tuning notes: {sorted(unknown_notes)}.")

        self._cache.clear()
        self._evaluations = 0
        rng = np.random.default_rng(self.config.random_seed)
        seed = self.codec.from_geometry()
        baseline = self._score(seed, selected_notes, retune=False)
        population = self._initial_population(seed, rng)

        for _ in range(self.config.generations):
            scores = [self._evaluate(candidate, selected_notes) for candidate in population]
            elite_indices = sorted(
                range(len(population)), key=lambda index: scores[index].rank
            )[: self.config.elite_count]
            next_population = [population[index].copy() for index in elite_indices]
            while len(next_population) < self.config.population_size:
                first = self._tournament(population, scores, rng)
                second = self._tournament(population, scores, rng)
                next_population.append(self._child(first, second, rng))
            population = np.asarray(next_population)

        best = min(
            (self._evaluate(candidate, selected_notes) for candidate in population),
            key=lambda score: score.rank,
        )
        if baseline.rank <= best.rank:
            best = baseline
        best_vector = best.vector
        bore, layout = self.codec.decode(best_vector)
        root_error = best.errors.get("Degree_1")
        root_satisfied = (
            None
            if root_error is None
            else bool(abs(root_error) <= ROOT_TUNING_TOLERANCE_CENTS)
        )
        return GeneticOptimizationResult(
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
    ).optimize(notes)
