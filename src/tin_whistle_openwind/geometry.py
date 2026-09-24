"""Reusable OpenWInD mouthpiece/pipe geometry building blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openwind import ImpedanceComputation, Player

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TEMPERATURE = 20.0


@dataclass
class WhistleInstrument:
    """Thin OpenWInD adapter exposing a `get_impedance` method for a bore profile."""

    bore: list[tuple[float, float]]
    radiation: str = "unflanged"
    temperature: float = DEFAULT_TEMPERATURE

    def get_impedance(self, frequencies: np.ndarray) -> np.ndarray:
        computation = ImpedanceComputation(
            frequencies=frequencies,
            main_bore=self.bore,
            unit="m",
            temperature=self.temperature,
            radiation_category=self.radiation,
            player=Player("UNITARY_FLOW"),
            compute_method="TMM",
            losses=True,
            nondim=False,
        )
        return np.asarray(computation.impedance, dtype=complex)


@dataclass
class MouthpieceGeometry:
    """OpenWInD bore points for a mouthpiece, in the mouthpiece's own bore radius only.

    The segments must NOT include the junction step to the attached pipe's radius;
    `WhistleGeometryFactory.build_bore` inserts that step for the given pipe.
    """

    name: str
    segments: list[tuple[float, float]]
    l_mouth: float  # calibrated acoustic extension added ahead of the attached pipe

    @property
    def exit_radius(self) -> float:
        """The mouthpiece's own bore radius at its junction with the pipe."""
        return self.segments[-1][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "segments": [list(point) for point in self.segments],
            "l_mouth_m": self.l_mouth,
        }

    def save(self, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{self.name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, name: str, data_dir: Path = DEFAULT_DATA_DIR) -> "MouthpieceGeometry":
        payload = json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))
        segments = [tuple(point) for point in payload["segments"]]
        return cls(name=payload["name"], segments=segments, l_mouth=payload["l_mouth_m"])


@dataclass
class PipeGeometry:
    """A cylindrical body pipe attached behind a mouthpiece."""

    name: str
    length_m: float
    radius_m: float


class WhistleGeometryFactory:
    """Combines a named mouthpiece with a pipe into an OpenWInD-ready WhistleInstrument."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        temperature: float = DEFAULT_TEMPERATURE,
        radiation: str = "unflanged",
    ):
        self.data_dir = data_dir
        self.temperature = temperature
        self.radiation = radiation

    def load_mouthpiece(self, name: str) -> MouthpieceGeometry:
        return MouthpieceGeometry.load(name, self.data_dir)

    def build_bore(
        self, mouthpiece: MouthpieceGeometry | str, pipe: PipeGeometry
    ) -> list[tuple[float, float]]:
        if isinstance(mouthpiece, str):
            mouthpiece = self.load_mouthpiece(mouthpiece)
        junction_position = mouthpiece.segments[-1][0]
        junction_step = (junction_position, pipe.radius_m)  # step from mouthpiece bore to pipe bore
        pipe_end = (mouthpiece.l_mouth + pipe.length_m, pipe.radius_m)
        return [*mouthpiece.segments, junction_step, pipe_end]

    def create_instrument(
        self, mouthpiece: MouthpieceGeometry | str, pipe: PipeGeometry
    ) -> WhistleInstrument:
        bore = self.build_bore(mouthpiece, pipe)
        return WhistleInstrument(bore=bore, radiation=self.radiation, temperature=self.temperature)
