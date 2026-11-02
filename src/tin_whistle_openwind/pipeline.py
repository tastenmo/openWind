"""Three-stage multi-key whistle design pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geometry import PipeGeometry, TaperedBoreGeometry, UndercutTonehole, WhistleGeometryFactory
from .keys import SUPPORTED_KEYS, WhistleKeyConfig, inner_radius_m, pipe_length_m, scale_mouthpiece_length_m
from .optimize_bore import BoreOptimizationResult, optimize_bore
from .optimize_toneholes import optimize_toneholes
from .optimize_undercut import UndercutOptimizationResult, optimize_undercut
from .optimizer import ToneholeLayout, ToneholeOptimizationResult
from .tuning import IRISH_FOLK_HYBRID, TuningProfile


@dataclass
class FinalWhistleDesign:
    """Serializable output of all three design stages."""

    key_name: str
    temperament_name: str
    bore: TaperedBoreGeometry
    holes: ToneholeLayout
    undercut: UndercutOptimizationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_name": self.key_name,
            "temperament": self.temperament_name,
            "bore": {"name": self.bore.name, "segments": [list(p) for p in self.bore.segments]},
            "holes": [
                {
                    "label": h.label,
                    "position_m": h.position_m,
                    "diameter_m": h.diameter_m,
                    "chimney_height_m": h.chimney_height_m,
                    "bevel_deg": h.bevel_deg,
                }
                for h in self.holes.holes
            ],
            "metadata": self.metadata,
        }


class WhistleDesignPipeline:
    """Orchestrate bore, tonehole and undercut stages for any supported key."""

    def __init__(
        self,
        key_name: str,
        temperament: TuningProfile = IRISH_FOLK_HYBRID,
        data_dir: Path | None = None,
    ) -> None:
        if key_name not in SUPPORTED_KEYS:
            raise KeyError(f"Unsupported key {key_name!r}; choose from {sorted(SUPPORTED_KEYS)}.")
        self.key_config: WhistleKeyConfig = SUPPORTED_KEYS[key_name]
        self.temperament = temperament
        self.geometry_factory = WhistleGeometryFactory(data_dir=data_dir) if data_dir else WhistleGeometryFactory()
        self.bore_result: BoreOptimizationResult | None = None
        self.tonehole_result: ToneholeOptimizationResult | None = None
        self.undercut_result: UndercutOptimizationResult | None = None

    def run_stage_1_bore_optimization(self) -> TaperedBoreGeometry:
        self.bore_result = optimize_bore(self.key_config)
        return self.bore_result.bore

    def run_stage_2_tonehole_optimization(self, bore: TaperedBoreGeometry) -> ToneholeLayout:
        self.tonehole_result = optimize_toneholes(self.key_config, bore, self.temperament)
        return self.tonehole_result.layout

    def run_stage_3_undercut_optimization(
        self, bore: TaperedBoreGeometry, holes: ToneholeLayout
    ) -> FinalWhistleDesign:
        self.undercut_result = optimize_undercut(holes)
        return FinalWhistleDesign(
            key_name=self.key_config.name,
            temperament_name=self.temperament.name,
            bore=bore,
            holes=self.undercut_result.layout,
            undercut=self.undercut_result,
            metadata={"root_note": self.key_config.root_note},
        )
