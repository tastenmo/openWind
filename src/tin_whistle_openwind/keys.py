"""Key definitions and acoustic scaling for multi-key tin whistle design."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WhistleKeyConfig:
    """Physical starting parameters for a whistle in a given key."""

    name: str                          # e.g. "Soprano D", "Low D"
    root_note: str                     # e.g. "D5", "D4"
    root_freq_12tet_hz: float          # e.g. D5=587.33
    default_inner_diameter_mm: float   # e.g. D5=14.4mm
    wall_thickness_mm: float           # e.g. 0.8mm PVC, 1.0-1.5mm low whistles
    approx_length_mm: float            # starting physical length approximation
    fork_note_name: str                # chromatic fork note label (e.g. "Fork_Minor7")
    octave_note_name: str              # octave note label (e.g. "Octave_Degree_1")


# Supported key registry.
SUPPORTED_KEYS: dict[str, WhistleKeyConfig] = {
    "Soprano_D": WhistleKeyConfig(
        name="Soprano High D",
        root_note="D5",
        root_freq_12tet_hz=587.33,
        default_inner_diameter_mm=14.4,
        wall_thickness_mm=0.8,
        approx_length_mm=263.61,
        fork_note_name="Fork_Minor7",
        octave_note_name="Octave_Degree_1",
    ),
    "Soprano_C": WhistleKeyConfig(
        name="Soprano C",
        root_note="C5",
        root_freq_12tet_hz=523.25,
        default_inner_diameter_mm=15.2,
        wall_thickness_mm=1.0,
        approx_length_mm=298.0,
        fork_note_name="Fork_Minor7",
        octave_note_name="Octave_Degree_1",
    ),
    "High_Bb": WhistleKeyConfig(
        name="High Bb",
        root_note="Bb4",
        root_freq_12tet_hz=466.16,
        default_inner_diameter_mm=16.2,
        wall_thickness_mm=1.0,
        approx_length_mm=335.0,
        fork_note_name="Fork_Minor7",
        octave_note_name="Octave_Degree_1",
    ),
    "Alto_A": WhistleKeyConfig(
        name="Alto A",
        root_note="A4",
        root_freq_12tet_hz=440.00,
        default_inner_diameter_mm=17.5,
        wall_thickness_mm=1.2,
        approx_length_mm=358.0,
        fork_note_name="Fork_Minor7",
        octave_note_name="Octave_Degree_1",
    ),
    "Low_D": WhistleKeyConfig(
        name="Low D",
        root_note="D4",
        root_freq_12tet_hz=293.66,
        default_inner_diameter_mm=24.0,
        wall_thickness_mm=1.5,
        approx_length_mm=530.0,
        fork_note_name="Fork_Minor7",
        octave_note_name="Octave_Degree_1",
    ),
}

# The only key with an experimentally calibrated mouthpiece (see data/MouthpieceDN16V2.json).
REFERENCE_KEY_NAME = "Soprano_D"
REFERENCE_L_MOUTH_M = 0.021991361950235867


def scale_mouthpiece_length_m(
    key_config: WhistleKeyConfig,
    reference_key: WhistleKeyConfig | None = None,
    reference_l_mouth_m: float = REFERENCE_L_MOUTH_M,
) -> float:
    """Scale the calibrated D5 mouthpiece acoustic length to another key.

    Acoustic length scales inversely with frequency (f * L = const for a given mode),
    so `l_mouth` for a new key is `reference_l_mouth * f_reference / f_key`.
    """
    reference_key = reference_key or SUPPORTED_KEYS[REFERENCE_KEY_NAME]
    return reference_l_mouth_m * reference_key.root_freq_12tet_hz / key_config.root_freq_12tet_hz


def inner_radius_m(key_config: WhistleKeyConfig) -> float:
    """The key's default inner bore radius, in meters."""
    return key_config.default_inner_diameter_mm / 2.0 / 1000.0


def pipe_length_m(key_config: WhistleKeyConfig) -> float:
    """The key's approximate physical pipe length, in meters."""
    return key_config.approx_length_mm / 1000.0
