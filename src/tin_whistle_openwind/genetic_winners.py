"""Persistence for compatible genetic-optimization winners."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_ARCHIVE_VERSION = 1
_MAX_WINNERS = 10


@contextmanager
def _archive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class WinnerArchive:
    """Keep the best ten distinct genomes for one exact scoring context."""

    def __init__(self, directory: Path, context: dict) -> None:
        canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.context = json.loads(canonical)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.path = directory / f"{digest}.json"
        self.lock_path = directory / f"{digest}.lock"

    def _read(self) -> list[tuple[tuple[float, ...], tuple[float, float]]]:
        try:
            with self.path.open(encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid genetic winner archive: {self.path}") from error
        if (
            not isinstance(data, dict)
            or data.get("version") != _ARCHIVE_VERSION
            or data.get("context") != self.context
            or not isinstance(data.get("winners"), list)
            or len(data["winners"]) > _MAX_WINNERS
        ):
            raise ValueError(f"Invalid genetic winner archive: {self.path}")
        winners = []
        for item in data["winners"]:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid genetic winner archive: {self.path}")
            genome = item.get("genome")
            rank = item.get("rank")
            if (
                not isinstance(genome, list)
                or len(genome) not in (
                    (31, 32) if self.context.get("genome_schema") == 2 else (20, 21)
                )
                or not isinstance(rank, list)
                or len(rank) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in genome + rank
                )
                or rank[0] < 0
                or (rank[1] < 0 and self.context.get("scoring_revision", 0) < 4)
            ):
                raise ValueError(f"Invalid genetic winner archive: {self.path}")
            winners.append((tuple(float(v) for v in genome), (float(rank[0]), float(rank[1]))))
        if len({genome for genome, _ in winners}) != len(winners):
            raise ValueError(f"Invalid genetic winner archive: {self.path}")
        return winners

    def load(self) -> list[tuple[float, ...]]:
        with _archive_lock(self.lock_path):
            return [genome for genome, _ in self._read()]

    def merge(
        self, winners: list[tuple[tuple[float, ...], tuple[float, float]]]
    ) -> None:
        with _archive_lock(self.lock_path):
            combined = dict(self._read())
            current: dict[tuple[float, ...], tuple[float, float]] = {}
            for genome, rank in winners:
                if genome not in current or rank < current[genome]:
                    current[genome] = rank
            combined.update(current)
            best = sorted(combined.items(), key=lambda item: (item[1], item[0]))[:_MAX_WINNERS]
            payload = {
                "version": _ARCHIVE_VERSION,
                "context": self.context,
                "winners": [{"genome": genome, "rank": rank} for genome, rank in best],
            }
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=self.path.name + ".", suffix=".tmp", delete=False,
                ) as file:
                    temporary = file.name
                    json.dump(payload, file, indent=2, allow_nan=False)
                    file.write("\n")
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
