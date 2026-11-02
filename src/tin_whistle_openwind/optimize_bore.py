"""Stage 1: bore taper and bell-end optimization."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import TaperedBoreGeometry
from .keys import WhistleKeyConfig, inner_radius_m, pipe_length_m


@dataclass
class BoreOptimizationResult:
    """Stage 1 result with the optimized starting bore profile."""

    bore: TaperedBoreGeometry
    octave_ratio: float


def optimize_bore(key_config: WhistleKeyConfig) -> BoreOptimizationResult:
    """Create the initial cylindrical bore and a neutral octave target.

    The actual octave fit is exposed as a separate result field so a measured
    impedance-based fit can replace this deterministic starting point later.
    """
    bore = TaperedBoreGeometry.cylindrical(
        name=f"{key_config.name} cylindrical bore",
        radius_m=inner_radius_m(key_config),
        length_m=pipe_length_m(key_config),
    )
    return BoreOptimizationResult(bore=bore, octave_ratio=2.0)
