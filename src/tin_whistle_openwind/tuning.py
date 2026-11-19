"""Key-agnostic tuning temperaments, scale degrees and fingering charts."""

from __future__ import annotations

from dataclasses import dataclass

from .keys import WhistleKeyConfig

# Diatonic interval offsets in cents relative to the key's root note.
# Root(0), M2(200), M3(400), P4(500), P5(700), M6(900), m7/fork(1000), M7(1100), octave(1200).
INTERVAL_OFFSETS_12TET: list[float] = [0, 200, 400, 500, 700, 900, 1000, 1100, 1200]

# Scale-degree labels, independent of key; the fingering chart below is keyed on these.
DEGREE_LABELS: list[str] = [
    "Degree_1",
    "Degree_2",
    "Degree_3",
    "Degree_4",
    "Degree_5",
    "Degree_6",
    "Fork_Minor7",
    "Degree_7",
    "Octave_Degree_1",
]

# Cent corrections per temperament, aligned with DEGREE_LABELS / INTERVAL_OFFSETS_12TET.
TEMPERAMENTS: dict[str, list[float]] = {
    "12-TET": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Irish_Folk_Hybrid": [0.0, +2.0, -8.0, 0.0, 0.0, -8.0, -6.0, -5.0, 0.0],
    "Just_Intonation": [0.0, +4.0, -14.0, -2.0, +2.0, -16.0, -10.0, -12.0, 0.0],
}

DEFAULT_TEMPERAMENT = "Irish_Folk_Hybrid"


@dataclass(frozen=True)
class TuningProfile:
    """A temperament represented by cent offsets from 12-TET."""

    name: str
    cent_offsets: dict[str, float]

    def get_target_frequency(self, note: str, base_12tet_hz: float) -> float:
        cent = self.cent_offsets.get(note, 0.0)
        return base_12tet_hz * (2.0 ** (cent / 1200.0))


IRISH_FOLK_HYBRID = TuningProfile(
    name="Irish Folk Hybrid (HB-trad / Copeland standard)",
    cent_offsets=dict(zip(DEGREE_LABELS, TEMPERAMENTS["Irish_Folk_Hybrid"])),
)
EQUAL_TEMPERAMENT = TuningProfile(
    name="12-TET (Equal Temperament)",
    cent_offsets=dict(zip(DEGREE_LABELS, TEMPERAMENTS["12-TET"])),
)
JUST_INTONATION = TuningProfile(
    name="Pure Just Intonation (Harmonic D-Major)",
    cent_offsets=dict(zip(DEGREE_LABELS, TEMPERAMENTS["Just_Intonation"])),
)

# Fingering charts (1 = closed hole, 0 = open hole), topological and thus key-independent.
# Hole order: [Hole 1 (top), Hole 2, Hole 3, Hole 4, Hole 5, Hole 6 (bottom)]
STANDARD_FINGERINGS: dict[str, list[int]] = {
    "Degree_1":        [1, 1, 1, 1, 1, 1],
    "Degree_2":        [1, 1, 1, 1, 1, 0],
    "Degree_3":        [1, 1, 1, 1, 0, 0],
    "Degree_4":        [1, 1, 1, 0, 0, 0],
    "Degree_5":        [1, 1, 0, 0, 0, 0],
    "Degree_6":        [1, 0, 0, 0, 0, 0],
    "Fork_Minor7":     [0, 1, 1, 0, 0, 0],  # standard fork fingering (Gabelgriff)
    "Degree_7":        [0, 0, 0, 0, 0, 0],
    "Octave_Degree_1": [0, 1, 1, 1, 1, 1],  # vented octave fingering (hole 1 open, all others closed)
}

# Upper-register GA targets omit the fork; the lower-register fork is retained.
SECOND_REGISTER_FINGERINGS = {
    f"Octave_{degree}": "Octave_Degree_1" if degree == "Degree_1" else degree
    for degree in DEGREE_LABELS if degree not in {"Octave_Degree_1", "Fork_Minor7"}
}


def generate_key_frequencies(
    key_config: WhistleKeyConfig,
    temperament: str | TuningProfile = DEFAULT_TEMPERAMENT,
) -> dict[str, float]:
    """Target frequencies (Hz) for all diatonic degrees + fork note, for a given key/temperament."""
    root_f = key_config.root_freq_12tet_hz
    scale_12tet = [root_f * (2.0 ** (cent / 1200.0)) for cent in INTERVAL_OFFSETS_12TET]
    if isinstance(temperament, TuningProfile):
        cent_modifiers = [temperament.cent_offsets[label] for label in DEGREE_LABELS]
    else:
        cent_modifiers = TEMPERAMENTS[temperament]

    return {
        label: base_f * (2.0 ** (mod / 1200.0))
        for label, base_f, mod in zip(DEGREE_LABELS, scale_12tet, cent_modifiers)
    }
