from __future__ import annotations

import json

import pytest

from tin_whistle_openwind.geometry import TaperedBoreGeometry, UndercutTonehole
from tin_whistle_openwind.keys import (
    SUPPORTED_KEYS,
    inner_radius_m,
    pipe_length_m,
    scale_mouthpiece_length_m,
)
from tin_whistle_openwind.pipeline import WhistleDesignPipeline
from tin_whistle_openwind.optimize_bore import optimize_bore
from tin_whistle_openwind.optimize_toneholes import optimize_toneholes
from tin_whistle_openwind.tuning import (
    IRISH_FOLK_HYBRID,
    STANDARD_FINGERINGS,
    generate_key_frequencies,
)


def test_supported_keys_have_positive_physical_parameters():
    assert set(SUPPORTED_KEYS) == {"Soprano_D", "Soprano_C", "High_Bb", "Alto_A", "Low_D"}
    for key in SUPPORTED_KEYS.values():
        assert key.root_freq_12tet_hz > 0
        assert inner_radius_m(key) > 0
        assert pipe_length_m(key) > 0


def test_mouthpiece_scale_preserves_reference_value():
    key = SUPPORTED_KEYS["Soprano_D"]
    assert scale_mouthpiece_length_m(key) == pytest.approx(0.021991361950235867)


def test_tuning_changes_frequency_by_cents():
    key = SUPPORTED_KEYS["Soprano_D"]
    targets = generate_key_frequencies(key, "Irish_Folk_Hybrid")
    assert targets["Degree_1"] == pytest.approx(587.33)
    assert targets["Degree_2"] > generate_key_frequencies(key, "12-TET")["Degree_2"]


def test_required_fork_and_vented_fingerings_are_present():
    assert STANDARD_FINGERINGS["Fork_Minor7"] == [0, 1, 1, 0, 0, 0]
    assert STANDARD_FINGERINGS["Octave_Degree_1"] == [0, 1, 1, 1, 1, 1]


def test_six_hole_reference_layout_excludes_lth():
    pipeline = WhistleDesignPipeline("Soprano_D", IRISH_FOLK_HYBRID)
    result = optimize_toneholes(
        pipeline.key_config,
        optimize_bore(pipeline.key_config).bore,
        IRISH_FOLK_HYBRID,
    )
    assert [hole.label for hole in result.layout.holes] == ["L1", "L2", "L3", "R1", "R2", "R3"]
    assert result.layout.diameters_m == pytest.approx(
        [0.0055, 0.0065, 0.006, 0.005, 0.008, 0.008], abs=1.1e-6
    )


def test_tapered_bore_and_undercut_validation():
    bore = TaperedBoreGeometry.with_bell_contraction("test", 0.0072, 0.26361, 0.0068)
    assert bore.length_m == pytest.approx(0.26361)
    assert bore.segments[-1][1] == pytest.approx(0.0068)
    hole = UndercutTonehole("hole1", 0.1, 0.006, 0.003, bevel_deg=15.0)
    assert hole.radius_out_m > hole.radius_in_m
    with pytest.raises(ValueError):
        UndercutTonehole("bad", 0.1, 0.006, 0.003, bevel_deg=16.0)


def test_pipeline_stage_outputs():
    pipeline = WhistleDesignPipeline("Soprano_D", IRISH_FOLK_HYBRID)
    bore = pipeline.run_stage_1_bore_optimization()
    holes = pipeline.run_stage_2_tonehole_optimization(bore)
    design = pipeline.run_stage_3_undercut_optimization(bore, holes)
    assert len(design.holes.holes) == 6
    assert design.key_name == "Soprano High D"
    assert design.to_dict()["temperament"] == IRISH_FOLK_HYBRID.name


def test_key_validation():
    with pytest.raises(KeyError):
        WhistleDesignPipeline("not-a-key")
