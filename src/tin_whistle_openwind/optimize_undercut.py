"""Stage 3: tonehole undercut and chimney bevel optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .optimizer import ToneholeLayout


@dataclass
class UndercutOptimizationResult:
    """Stage 3 result containing the final undercut-adjusted toneholes."""

    layout: ToneholeLayout


def optimize_undercut(layout: ToneholeLayout, bevel_deg: float = 0.0) -> UndercutOptimizationResult:
    """Apply a bounded chimney bevel recommendation to all holes.

    This deterministic baseline keeps external hole positions unchanged; a measured
    second-octave objective can optimize `bevel_deg` in a subsequent calibration.
    """
    holes = [
        replace(hole, bevel_deg=bevel_deg)
        for hole in layout.holes
    ]
    return UndercutOptimizationResult(ToneholeLayout(holes))
