"""Stage 2: tonehole placement and sizing optimization entry point."""

from __future__ import annotations

from typing import Sequence

from .geometry import MouthpieceGeometry, TaperedBoreGeometry, WhistleInstrument
from .keys import WhistleKeyConfig
from .optimizer import (
    ACTIVE_HOLE_LABELS,
    REFERENCE_HOLE_DIAMETERS_M,
    ToneholeOptimizationResult,
    ToneholeOptimizer,
)
from .tuning import IRISH_FOLK_HYBRID, TuningProfile


def optimize_toneholes(
    key_config: WhistleKeyConfig,
    bore: TaperedBoreGeometry | WhistleInstrument | list[list[float]],
    tuning: TuningProfile = IRISH_FOLK_HYBRID,
    initial_positions_m: Sequence[float] | None = None,
    initial_diameters_m: Sequence[float] | None = None,
    mouthpiece: MouthpieceGeometry | None = None,
) -> ToneholeOptimizationResult:
    """Fit holes and pipe length, including the calibrated mouthpiece.

    For pipe-only bores, use ``result.bore`` for subsequent simulation/export:
    the Stage 1 length changes to account for the closed tonehole cavities.
    """
    if mouthpiece is None and isinstance(bore, TaperedBoreGeometry):
        mouthpiece = MouthpieceGeometry.load("MouthpieceDN16V2")
    pipe_length_m = (
        bore.length_m if isinstance(bore, TaperedBoreGeometry)
        else key_config.approx_length_mm / 1000.0
    )
    if initial_positions_m is None:
        reference_from_end_m = (0.1465, 0.1267, 0.1076, 0.0845, 0.0667, 0.0391)
        initial_positions_m = [
            pipe_length_m - from_end for from_end in reference_from_end_m
        ]
        initial_positions_m = [
            max(position, initial_positions_m[index - 1] + 0.0185)
            if index > 0 else position
            for index, position in enumerate(initial_positions_m)
        ]
    if initial_diameters_m is None:
        initial_diameters_m = REFERENCE_HOLE_DIAMETERS_M
    return ToneholeOptimizer(key_config, tuning, bore, mouthpiece=mouthpiece).optimize(
        initial_positions_m, initial_diameters_m
    )
