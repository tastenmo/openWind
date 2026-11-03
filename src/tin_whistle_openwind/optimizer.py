"""Tonehole simulation and constrained optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.signal import find_peaks

from openwind import ImpedanceComputation, Player
from openwind import InstrumentGeometry, InstrumentPhysics
from openwind.inversion import InverseFrequentialResponse

from .geometry import (
    HOLES_VALVES_HEADER,
    MouthpieceGeometry,
    TaperedBoreGeometry,
    UndercutTonehole,
    WhistleGeometryFactory,
    WhistleInstrument,
)
from .keys import WhistleKeyConfig
from .tuning import DEGREE_LABELS, STANDARD_FINGERINGS, TuningProfile, generate_key_frequencies

MIN_HOLE_SPACING_M = 0.018
MIN_ADJACENT_HOLE_SPACINGS_M = (
    MIN_HOLE_SPACING_M, MIN_HOLE_SPACING_M, 0.0,
    MIN_HOLE_SPACING_M, MIN_HOLE_SPACING_M,
)
MIN_HOLE_DIAMETER_M = 0.005
MAX_HOLE_DIAMETER_M = 0.008
MIN_BOTTOM_END_DISTANCE_M = 0.030
GEOMETRY_TOLERANCE_M = 1e-9
ROOT_TUNING_TOLERANCE_CENTS = 0.1
ACTIVE_HOLE_LABELS: tuple[str, ...] = ("L1", "L2", "L3", "R1", "R2", "R3")
REFERENCE_HOLE_DIAMETERS_M: tuple[float, ...] = (0.0055, 0.0065, 0.0060, 0.0050, 0.0080, 0.0080)


@dataclass
class ToneholeLayout:
    """External tonehole positions and diameters, ordered from top to bottom."""

    holes: list[UndercutTonehole]

    @property
    def positions_m(self) -> np.ndarray:
        return np.array([hole.position_m for hole in self.holes], dtype=float)

    @property
    def diameters_m(self) -> np.ndarray:
        return np.array([hole.diameter_m for hole in self.holes], dtype=float)

    def validate(self, pipe_length_m: float) -> None:
        positions = self.positions_m
        diameters = self.diameters_m
        if len(positions) != 6:
            raise ValueError("A six-hole whistle layout is required.")
        if np.any(np.diff(positions) <= 0):
            raise ValueError("Toneholes must remain ordered from L1 to R3.")
        if np.any(
            np.diff(positions)
            < np.asarray(MIN_ADJACENT_HOLE_SPACINGS_M) - GEOMETRY_TOLERANCE_M
        ):
            raise ValueError("Adjacent toneholes within each hand must be at least 18 mm apart.")
        if np.any((diameters < MIN_HOLE_DIAMETER_M) | (diameters > MAX_HOLE_DIAMETER_M)):
            raise ValueError("Tonehole diameters must be between 5 and 8 mm.")
        if pipe_length_m - positions[-1] < MIN_BOTTOM_END_DISTANCE_M - GEOMETRY_TOLERANCE_M:
            raise ValueError("The bottom hole must be at least 30 mm from the pipe end.")

    def to_openwind_rows(self) -> list[list[object]]:
        """Convert holes to OpenWInD's header-plus-rows format."""
        return [HOLES_VALVES_HEADER, *[hole.to_row() for hole in self.holes]]

    def to_vector(self) -> np.ndarray:
        return np.r_[self.positions_m, self.diameters_m]


@dataclass
class ToneholeOptimizationResult:
    layout: ToneholeLayout
    result: OptimizeResult
    cent_errors: dict[str, float]
    initial_cost: float
    final_cost: float
    improvement: float
    improved: bool
    bore: TaperedBoreGeometry | WhistleInstrument | list[list[float]]


@dataclass
class InverseToneholeOptimizationResult:
    """Result of OpenWInD's measured-impedance inverse optimization."""

    result: object
    geometry: InstrumentGeometry
    inverse: InverseFrequentialResponse
    initial_cost: float
    final_cost: float

    @property
    def improved(self) -> bool:
        return self.final_cost < self.initial_cost


@dataclass
class AdmittanceResponse:
    frequencies_hz: np.ndarray
    impedance: np.ndarray
    admittance: np.ndarray
    peak_frequency_hz: float


def fingering_chart_rows(
    fingerings: dict[str, list[int]] = STANDARD_FINGERINGS,
    hole_labels: Sequence[str] | None = None,
) -> list[list[object]]:
    """Build OpenWInD chart rows; OpenWInD uses 'x' for closed and 'o' for open."""
    hole_labels = list(hole_labels or ACTIVE_HOLE_LABELS)
    if len(hole_labels) != 6:
        raise ValueError("Six hole labels are required.")
    rows: list[list[object]] = [["label", *fingerings.keys()]]
    for index, label in enumerate(hole_labels):
        rows.append([label, *["x" if values[index] else "o" for values in fingerings.values()]])
    return rows


def find_play_resonance(
    impedance: np.ndarray,
    frequencies_hz: np.ndarray,
    target_hz: float,
) -> float:
    """Find the nearest admittance peak to a target frequency."""
    admittance = np.divide(1.0, np.abs(impedance), out=np.zeros_like(np.abs(impedance)), where=np.abs(impedance) > 0)
    peak_indices, _ = find_peaks(admittance)
    if len(peak_indices) == 0:
        return float(frequencies_hz[np.argmax(admittance)])
    nearest = int(peak_indices[np.argmin(np.abs(frequencies_hz[peak_indices] - target_hz))])
    if nearest == 0 or nearest == len(admittance) - 1:
        return float(frequencies_hz[nearest])

    # Sub-grid quadratic interpolation avoids a staircase objective on the solver grid.
    left, center, right = admittance[nearest - 1:nearest + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < np.finfo(float).eps:
        return float(frequencies_hz[nearest])
    offset = 0.5 * (left - right) / denominator
    step = frequencies_hz[nearest + 1] - frequencies_hz[nearest]
    return float(frequencies_hz[nearest] + offset * step)


class OpenWInDToneholeModel:
    """Reusable OpenWInD wrapper for one bore and a six-hole fingering chart."""

    def __init__(
        self,
        bore: TaperedBoreGeometry | WhistleInstrument | list[list[float]],
        layout: ToneholeLayout,
        temperature: float = 20.0,
        radiation: str = "unflanged",
        hole_position_offset_m: float = 0.0,
    ) -> None:
        self.bore = bore
        self.layout = layout
        self.temperature = temperature
        self.radiation = radiation
        self.hole_position_offset_m = hole_position_offset_m

    def _main_bore(self) -> object:
        if isinstance(self.bore, TaperedBoreGeometry):
            return self.bore.segments
        if isinstance(self.bore, WhistleInstrument):
            return self.bore.bore
        return self.bore

    def impedance(
        self,
        frequencies_hz: np.ndarray,
        note: str,
        fingerings: dict[str, list[int]] = STANDARD_FINGERINGS,
    ) -> np.ndarray:
        layout = ToneholeLayout(
            [
                UndercutTonehole(
                    hole.label,
                    hole.position_m + self.hole_position_offset_m,
                    hole.diameter_m,
                    hole.chimney_height_m,
                    hole.bevel_deg,
                )
                for hole in self.layout.holes
            ]
        )
        computation = ImpedanceComputation(
            frequencies_hz,
            self._main_bore(),
            layout.to_openwind_rows(),
            fingering_chart_rows(fingerings),
            unit="m",
            player=Player("UNITARY_FLOW"),
            temperature=self.temperature,
            radiation_category=self.radiation,
            compute_method="TMM",
            losses=True,
            nondim=False,
            note=note,
        )
        return np.asarray(computation.impedance, dtype=complex)

    def response(
        self,
        note: str,
        target_hz: float,
        f_min: float = 250.0,
        f_max: float = 2500.0,
        n_points: int = 900,
    ) -> AdmittanceResponse:
        frequencies = np.linspace(f_min, f_max, n_points)
        impedance = self.impedance(frequencies, note)
        admittance = np.divide(1.0, np.abs(impedance), out=np.zeros_like(np.abs(impedance)), where=np.abs(impedance) > 0)
        peak = find_play_resonance(impedance, frequencies, target_hz)
        return AdmittanceResponse(frequencies, impedance, admittance, peak)


class InverseToneholeOptimizer:
    """Fit six tonehole positions/radii to measured OpenWInD impedance targets.

    Target arrays must already be normalized by the entrance characteristic
    impedance, as required by ``InverseFrequentialResponse``.
    """

    def __init__(
        self,
        bore: TaperedBoreGeometry | WhistleInstrument | list[list[float]],
        layout: ToneholeLayout,
        temperature: float = 20.0,
        radiation: str = "unflanged",
        losses: bool = True,
    ) -> None:
        self.bore = bore
        self.layout = layout
        self.temperature = temperature
        self.radiation = radiation
        self.losses = losses

    def _main_bore(self) -> object:
        if isinstance(self.bore, TaperedBoreGeometry):
            return self.bore.segments
        if isinstance(self.bore, WhistleInstrument):
            return self.bore.bore
        return self.bore

    def _inverse_holes(self) -> list[list[object]]:
        rows: list[list[object]] = [["label", "position", "radius", "chimney"]]
        for hole in self.layout.holes:
            rows.append([
                hole.label,
                f"~{hole.position_m}",
                f"~{hole.radius_in_m}",
                hole.chimney_height_m,
            ])
        return rows

    def fit(
        self,
        frequencies_hz: np.ndarray,
        target_impedances_normalized: list[np.ndarray],
        notes: list[str],
        algorithm: str = "SLSQP",
        max_iter: int = 100,
        min_hole_spacing_m: float = MIN_HOLE_SPACING_M,
    ) -> InverseToneholeOptimizationResult:
        if len(target_impedances_normalized) != len(notes):
            raise ValueError("One normalized target impedance array is required per note.")
        if any(np.asarray(target).shape[-1] != len(frequencies_hz) for target in target_impedances_normalized):
            raise ValueError("Target impedance arrays must match frequencies_hz in length.")

        fingerings = fingering_chart_rows(hole_labels=ACTIVE_HOLE_LABELS)
        geometry = InstrumentGeometry(self._main_bore(), self._inverse_holes(), fingerings)
        for index, (first, second) in enumerate(
            zip(ACTIVE_HOLE_LABELS, ACTIVE_HOLE_LABELS[1:])
        ):
            geometry.constrain_2_holes_distance(
                first, second, Lmin=1e-4 if index == 2 else min_hole_spacing_m
            )
        physics = InstrumentPhysics(
            geometry,
            self.temperature,
            player=Player("UNITARY_FLOW"),
            losses=self.losses,
            radiation_category=self.radiation,
        )
        inverse = InverseFrequentialResponse(
            physics,
            frequencies_hz,
            target_impedances_normalized,
            observable="impedance",
            notes=notes,
            compute_method="TMM",
        )
        initial_cost = float(inverse.get_cost_grad_hessian()[0])
        result = inverse.optimize_freq_model(
            iter_detailed=True,
            algorithm=algorithm,
            max_iter=max_iter,
        )
        final_cost = float(inverse.get_cost_grad_hessian()[0])
        return InverseToneholeOptimizationResult(
            result=result,
            geometry=geometry,
            inverse=inverse,
            initial_cost=initial_cost,
            final_cost=final_cost,
        )


class ToneholeOptimizer:
    """Fit hole positions/diameters and pipe length to admittance peak targets.

    Pipe-only ``TaperedBoreGeometry`` inputs allow length adjustment within 5%
    of the starting length, keeping all radii and the terminal taper fixed.
    Assembled instruments and raw bore lists retain their fixed geometry.
    """

    def __init__(
        self,
        key_config: WhistleKeyConfig,
        tuning: TuningProfile,
        bore: TaperedBoreGeometry | WhistleInstrument | list[list[float]],
        temperature: float = 20.0,
        mouthpiece: MouthpieceGeometry | None = None,
    ) -> None:
        self.key_config = key_config
        self.tuning = tuning
        self.bore = bore
        self.temperature = temperature
        self.mouthpiece = mouthpiece
        self.pipe_length_m = (
            bore.length_m if isinstance(bore, TaperedBoreGeometry)
            else key_config.approx_length_mm / 1000.0
        )
        self.acoustic_bore = bore
        if mouthpiece is not None and isinstance(bore, TaperedBoreGeometry):
            factory = WhistleGeometryFactory(temperature=temperature)
            self.acoustic_bore = factory.create_tapered_instrument(mouthpiece, bore)

    def _optimized_bore(
        self, vector: np.ndarray
    ) -> TaperedBoreGeometry | WhistleInstrument | list[list[float]]:
        if isinstance(self.bore, TaperedBoreGeometry) and len(vector) == 13:
            return self.bore.with_length(float(vector[12]))
        return self.bore

    def _model(
        self,
        layout: ToneholeLayout,
        bore: TaperedBoreGeometry | WhistleInstrument | list[list[float]] | None = None,
    ) -> OpenWInDToneholeModel:
        acoustic_bore = self.acoustic_bore
        if bore is not None:
            acoustic_bore = bore
            if self.mouthpiece is not None and isinstance(bore, TaperedBoreGeometry):
                acoustic_bore = WhistleGeometryFactory(
                    temperature=self.temperature
                ).create_tapered_instrument(self.mouthpiece, bore)
        return OpenWInDToneholeModel(
            acoustic_bore,
            layout,
            self.temperature,
            hole_position_offset_m=self.mouthpiece.l_mouth if self.mouthpiece else 0.0,
        )

    def targets_hz(self) -> dict[str, float]:
        return generate_key_frequencies(self.key_config, "12-TET")

    def _layout(self, vector: np.ndarray) -> ToneholeLayout:
        holes = [
            UndercutTonehole(ACTIVE_HOLE_LABELS[i], vector[i], vector[i + 6], 0.003)
            for i in range(len(ACTIVE_HOLE_LABELS))
        ]
        return ToneholeLayout(holes)

    def _cent_error(self, model: OpenWInDToneholeModel, note: str) -> float:
        target = self.tuning.get_target_frequency(note, self.targets_hz()[note])
        response = model.response(
            note,
            target,
            f_min=0.75 * target,
            f_max=1.25 * target,
            n_points=300,
        )
        return float(1200.0 * np.log2(response.peak_frequency_hz / target))

    def objective(self, vector: np.ndarray, notes: Sequence[str]) -> float:
        try:
            layout = self._layout(vector)
            model = self._model(layout, self._optimized_bore(vector))
            return sum(self._cent_error(model, note) ** 2 for note in notes)
        except (ValueError, RuntimeError, IndexError, KeyError):
            return 1e12

    def optimize(
        self,
        initial_positions_m: Sequence[float],
        initial_diameters_m: Sequence[float],
        notes: Sequence[str] | None = None,
    ) -> ToneholeOptimizationResult:
        notes = list(notes or DEGREE_LABELS)
        x0 = np.r_[initial_positions_m, initial_diameters_m]
        pipe_length_m = self.pipe_length_m
        optimize_length = isinstance(self.bore, TaperedBoreGeometry)
        max_pipe_length_m = 1.05 * pipe_length_m if optimize_length else pipe_length_m
        bounds = [
            (
                MIN_HOLE_SPACING_M + sum(MIN_ADJACENT_HOLE_SPACINGS_M[:index]),
                max_pipe_length_m - MIN_BOTTOM_END_DISTANCE_M,
            )
            for index in range(6)
        ]
        bounds += [(MIN_HOLE_DIAMETER_M, MAX_HOLE_DIAMETER_M)] * 6
        diameter_margin = 1e-6
        x0[6:] = np.clip(
            x0[6:],
            MIN_HOLE_DIAMETER_M + diameter_margin,
            MAX_HOLE_DIAMETER_M - diameter_margin,
        )
        self._layout(x0).validate(pipe_length_m)
        if optimize_length:
            x0 = np.r_[x0, pipe_length_m]
            bounds.append((0.95 * pipe_length_m, max_pipe_length_m))
        initial_cost = self.objective(x0, notes)
        parameter_scale = 1000.0
        scaled_x0 = x0 * parameter_scale
        scaled_bounds = [
            (lower * parameter_scale, upper * parameter_scale)
            for lower, upper in bounds
        ]
        constraints = [
            {
                "type": "ineq",
                "fun": lambda vector, i=i: (
                    vector[i + 1]
                    - vector[i]
                    - (MIN_ADJACENT_HOLE_SPACINGS_M[i] + 1e-4) * parameter_scale
                ),
            }
            for i in range(5)
        ]
        if optimize_length:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda vector: (
                        vector[12] - vector[5]
                        - (MIN_BOTTOM_END_DISTANCE_M + 1e-4) * parameter_scale
                    ),
                }
            )
            if "Degree_1" in notes:
                def closed_root_error(scaled_vector: np.ndarray) -> float:
                    vector = scaled_vector / parameter_scale
                    model = self._model(self._layout(vector), self._optimized_bore(vector))
                    return self._cent_error(model, "Degree_1")

                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda vector: (
                            ROOT_TUNING_TOLERANCE_CENTS - abs(closed_root_error(vector))
                        ),
                    }
                )
        result = minimize(
            lambda vector: self.objective(vector / parameter_scale, notes) / 1000.0,
            scaled_x0,
            method="SLSQP",
            bounds=scaled_bounds,
            constraints=constraints,
            options={
                "ftol": 1e-5,
                "maxiter": 100 if optimize_length else 50,
                "eps": 0.001 if optimize_length else 0.01,
                "disp": False,
            },
        )
        optimized_vector = result.x / parameter_scale
        final_cost = self.objective(optimized_vector, notes)
        if not result.success:
            raise RuntimeError(f"Tonehole optimization failed: {result.message}")
        layout = self._layout(optimized_vector)
        optimized_bore = self._optimized_bore(optimized_vector)
        layout.validate(
            optimized_bore.length_m
            if isinstance(optimized_bore, TaperedBoreGeometry) else pipe_length_m
        )
        result.x = optimized_vector
        result.fun = final_cost
        if result.jac is not None:
            result.jac = np.asarray(result.jac) * parameter_scale * 1000.0
        model = self._model(layout, optimized_bore)
        cent_errors = {note: self._cent_error(model, note) for note in notes}
        improvement = float(initial_cost - final_cost)
        improved = improvement > max(1e-6, abs(initial_cost) * 1e-9)
        return ToneholeOptimizationResult(
            layout=layout,
            result=result,
            cent_errors=cent_errors,
            initial_cost=initial_cost,
            final_cost=final_cost,
            improvement=improvement,
            improved=improved,
            bore=optimized_bore,
        )
