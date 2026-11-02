"""Stage 1: pipe length, bore taper and endhole optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.signal import find_peaks

from .geometry import (
    DEFAULT_TEMPERATURE,
    MouthpieceGeometry,
    TaperedBoreGeometry,
    WhistleGeometryFactory,
)
from .keys import WhistleKeyConfig, pipe_length_m


@dataclass
class LengthEndholeOptimizationResult:
    """Stage 1 result with the optimized pipe length and endhole."""

    bore: TaperedBoreGeometry
    octave_ratio: float
    initial_frequencies_hz: tuple[float, float]
    optimized_frequencies_hz: tuple[float, float]
    initial_cost: float
    final_cost: float
    result: OptimizeResult


def _resonance_frequency(
    frequencies_hz: np.ndarray,
    impedance: np.ndarray,
    target_hz: float,
    low_hz: float,
    high_hz: float,
) -> float:
    admittance = np.divide(
        1.0,
        np.abs(impedance),
        out=np.zeros_like(np.abs(impedance)),
        where=np.abs(impedance) > 0,
    )
    in_window = np.flatnonzero((frequencies_hz >= low_hz) & (frequencies_hz <= high_hz))
    if len(in_window) < 3 or not np.all(np.isfinite(admittance[in_window])):
        raise RuntimeError("The impedance sweep did not produce finite resonance data.")

    peak_indices, _ = find_peaks(admittance)
    candidates = peak_indices[
        (frequencies_hz[peak_indices] >= low_hz) & (frequencies_hz[peak_indices] <= high_hz)
    ]
    if len(candidates):
        index = int(candidates[np.argmin(np.abs(frequencies_hz[candidates] - target_hz))])
    else:
        index = int(in_window[np.argmax(admittance[in_window])])

    if index == 0 or index == len(admittance) - 1:
        return float(frequencies_hz[index])
    left, center, right = admittance[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < np.finfo(float).eps:
        return float(frequencies_hz[index])
    offset = 0.5 * (left - right) / denominator
    step = frequencies_hz[index + 1] - frequencies_hz[index]
    return float(frequencies_hz[index] + offset * step)


def optimize_length_endhole(
    key_config: WhistleKeyConfig,
    mouthpiece: MouthpieceGeometry | None = None,
    starting_bore_radius_m: float | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LengthEndholeOptimizationResult:
    """Fit pipe length and terminal radius with a fixed 20 mm end taper.

    By default the pipe starts at the mouthpiece exit radius. Set
    ``starting_bore_radius_m`` to use a different constant main-pipe radius.
    """
    mouthpiece = mouthpiece or MouthpieceGeometry.load("MouthpieceDN16V2")
    starting_radius_m = (
        mouthpiece.exit_radius
        if starting_bore_radius_m is None
        else starting_bore_radius_m
    )
    if starting_radius_m <= 0.0:
        raise ValueError("starting_bore_radius_m must be positive.")
    root_target_hz = key_config.root_freq_12tet_hz
    octave_target_hz = 2.0 * root_target_hz
    frequencies_hz = np.linspace(0.65 * root_target_hz, 2.4 * root_target_hz, 1400)
    factory = WhistleGeometryFactory(temperature=temperature)
    initial_pipe_length_m = pipe_length_m(key_config)
    taper_length_m = 0.020

    def make_bore(parameters: np.ndarray) -> TaperedBoreGeometry:
        length_m, end_radius_m = parameters
        return TaperedBoreGeometry(
            name=f"{key_config.name} optimized bore",
            segments=[
                (0.0, starting_radius_m),
                (float(length_m - taper_length_m), starting_radius_m),
                (float(length_m), float(end_radius_m)),
            ],
        )

    def get_resonances(parameters: np.ndarray) -> tuple[float, float]:
        instrument = factory.create_tapered_instrument(mouthpiece, make_bore(parameters))
        impedance = instrument.get_impedance(frequencies_hz)
        root_hz = _resonance_frequency(
            frequencies_hz,
            impedance,
            root_target_hz,
            0.75 * root_target_hz,
            1.3 * root_target_hz,
        )
        octave_hz = _resonance_frequency(
            frequencies_hz,
            impedance,
            octave_target_hz,
            1.5 * root_target_hz,
            2.4 * root_target_hz,
        )
        return root_hz, octave_hz

    def residuals(parameters: np.ndarray) -> np.ndarray:
        root_hz, octave_hz = get_resonances(parameters)

        # 1. Absoluter Fehler des Grundtons (wird hauptsächlich von L_total gesteuert)
        root_error = 1200.0 * np.log2(root_hz / root_target_hz)

        # 2. Fehler des Oktavintervalls in Cents (wird hauptsächlich von end_radius gesteuert)
        target_interval = 1200.0 * np.log2(octave_target_hz / root_target_hz)
        calc_interval = 1200.0 * np.log2(octave_hz / root_hz)
        interval_error = calc_interval - target_interval

        # Intervall-Fehler höher gewichten (z. B. Faktor 2.0)
        return np.array([
            1.0 * root_error,
            2.0 * interval_error
        ])

    initial_parameters = np.array([initial_pipe_length_m, 0.95 * starting_radius_m])
    initial_frequencies = get_resonances(initial_parameters)
    initial_residuals = residuals(initial_parameters)
    initial_cost = float(initial_residuals @ initial_residuals)
    result = least_squares(
        residuals,
        initial_parameters,
        bounds=(
            [max(0.75 * initial_pipe_length_m, taper_length_m), 0.75 * starting_radius_m],
            [1.25 * initial_pipe_length_m, 1.0 * starting_radius_m],
        ),
        x_scale=[initial_pipe_length_m, starting_radius_m],
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=100,
    )
    if not result.success:
        raise RuntimeError(f"Bore optimization failed: {result.message}")

    optimized_frequencies = get_resonances(result.x)
    final_cost = float(result.fun @ result.fun)
    return LengthEndholeOptimizationResult(
        bore=make_bore(result.x),
        octave_ratio=optimized_frequencies[1] / optimized_frequencies[0],
        initial_frequencies_hz=initial_frequencies,
        optimized_frequencies_hz=optimized_frequencies,
        initial_cost=initial_cost,
        final_cost=final_cost,
        result=result,
    )


def optimize_bore(
    key_config: WhistleKeyConfig,
    mouthpiece: MouthpieceGeometry | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LengthEndholeOptimizationResult:
    """Compatibility wrapper for the Stage 1 length/endhole optimization."""
    return optimize_length_endhole(key_config, mouthpiece, temperature=temperature)


BoreOptimizationResult = LengthEndholeOptimizationResult
