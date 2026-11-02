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

    def build_tapered_bore(
        self, mouthpiece: MouthpieceGeometry | str, bore: "TaperedBoreGeometry"
    ) -> list[tuple[float, float]]:
        """Combine a mouthpiece with a (possibly tapered/belled) main bore profile."""
        if isinstance(mouthpiece, str):
            mouthpiece = self.load_mouthpiece(mouthpiece)
        junction_position = mouthpiece.segments[-1][0]
        junction_step = (junction_position, bore.segments[0][1])
        offset_segments = [
            (mouthpiece.l_mouth + position, radius) for position, radius in bore.segments[1:]
        ]
        return [*mouthpiece.segments, junction_step, *offset_segments]

    def create_tapered_instrument(
        self, mouthpiece: MouthpieceGeometry | str, bore: "TaperedBoreGeometry"
    ) -> WhistleInstrument:
        bore_points = self.build_tapered_bore(mouthpiece, bore)
        return WhistleInstrument(bore=bore_points, radiation=self.radiation, temperature=self.temperature)


@dataclass
class TaperedBoreGeometry:
    """A segmented (optionally tapered/belled) main bore, relative to the pipe's own start.

    `segments` is a list of (position_m, radius_m) pairs; the first entry must be at
    position 0.0 and set the nominal (cylindrical) bore radius at the mouthpiece junction.
    """

    name: str
    segments: list[tuple[float, float]]

    @property
    def length_m(self) -> float:
        return self.segments[-1][0]

    def with_length(self, length_m: float) -> "TaperedBoreGeometry":
        """Resize the pipe while preserving radii and the terminal taper length."""
        terminal_length = (
            self.segments[-1][0] - self.segments[-2][0]
            if len(self.segments) > 2 else 0.0
        )
        if not np.isfinite(length_m) or length_m <= terminal_length:
            raise ValueError("Pipe length must exceed the terminal taper length.")
        scale = (length_m - terminal_length) / (self.length_m - terminal_length)
        return TaperedBoreGeometry(
            name=self.name,
            segments=[
                (position * scale, radius) for position, radius in self.segments[:-1]
            ] + [(length_m, self.segments[-1][1])],
        )

    @classmethod
    def cylindrical(cls, name: str, radius_m: float, length_m: float) -> "TaperedBoreGeometry":
        return cls(name=name, segments=[(0.0, radius_m), (length_m, radius_m)])

    @classmethod
    def with_bell_contraction(
        cls,
        name: str,
        radius_m: float,
        length_m: float,
        bell_radius_m: float,
        taper_start_fraction: float = 0.85,
    ) -> "TaperedBoreGeometry":
        """A cylindrical bore with a reverse-conical taper into a bell end-contraction."""
        taper_start_m = length_m * taper_start_fraction
        return cls(
            name=name,
            segments=[
                (0.0, radius_m),
                (taper_start_m, radius_m),
                (length_m, bell_radius_m),
            ],
        )


@dataclass
class UndercutTonehole:
    """A tonehole with an optional inner chimney bevel (undercut), 0-15 degrees."""

    label: str
    position_m: float
    diameter_m: float
    chimney_height_m: float
    bevel_deg: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.bevel_deg <= 15.0:
            raise ValueError(f"bevel_deg must be within [0, 15], got {self.bevel_deg}.")

    @property
    def radius_in_m(self) -> float:
        """Chimney radius at the bore wall (unaffected by undercutting)."""
        return self.diameter_m / 2.0

    @property
    def radius_out_m(self) -> float:
        """Chimney radius at the outer (finger) side, flared by the undercut bevel."""
        return self.radius_in_m + self.chimney_height_m * np.tan(np.radians(self.bevel_deg))

    def to_row(self) -> list[Any]:
        """An OpenWInD `holes_valves` data row (conical chimney via radius/radius_out)."""
        return [
            "hole",
            self.label,
            self.position_m,
            self.radius_in_m,
            self.radius_out_m,
            self.chimney_height_m,
        ]


HOLES_VALVES_HEADER: list[str] = ["variety", "label", "position", "radius", "radius_out", "length"]
