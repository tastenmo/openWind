from __future__ import annotations

import json
import math

import pytest

from tin_whistle_openwind.geometry import TaperedBoreGeometry, UndercutTonehole
from tin_whistle_openwind.keys import (
    SUPPORTED_KEYS,
    inner_radius_m,
    pipe_length_m,
    scale_mouthpiece_length_m,
)
from tin_whistle_openwind.pipeline import WhistleDesignPipeline
from tin_whistle_openwind.optimize_bore import optimize_length_endhole
from tin_whistle_openwind.optimizer import ACTIVE_HOLE_LABELS, REFERENCE_HOLE_DIAMETERS_M
from tin_whistle_openwind.reporting import drilling_chart_rows, fingering_tuning_rows
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


def test_length_endhole_optimizer_defaults_to_mouthpiece_bore():
    result = optimize_length_endhole(SUPPORTED_KEYS["Soprano_D"])

    assert result.bore.segments[0][1] == pytest.approx(0.0065)
    assert result.bore.segments[1][1] == pytest.approx(0.0065)
    assert result.bore.segments[-1][0] - result.bore.segments[-2][0] == pytest.approx(0.020)


def test_length_endhole_optimizer_accepts_custom_starting_bore():
    result = optimize_length_endhole(
        SUPPORTED_KEYS["Soprano_D"],
        starting_bore_radius_m=0.007,
    )

    assert result.bore.segments[0][1] == pytest.approx(0.007)
    assert result.bore.segments[1][1] == pytest.approx(0.007)
    assert result.bore.segments[-1][0] - result.bore.segments[-2][0] == pytest.approx(0.020)


def test_length_endhole_optimizer_rejects_nonpositive_starting_bore():
    with pytest.raises(ValueError, match="starting_bore_radius_m"):
        optimize_length_endhole(
            SUPPORTED_KEYS["Soprano_D"],
            starting_bore_radius_m=0.0,
        )


def test_tuning_changes_frequency_by_cents():
    key = SUPPORTED_KEYS["Soprano_D"]
    targets = generate_key_frequencies(key, "Irish_Folk_Hybrid")
    assert targets["Degree_1"] == pytest.approx(587.33)
    assert targets["Degree_2"] > generate_key_frequencies(key, "12-TET")["Degree_2"]


def test_required_fork_and_vented_fingerings_are_present():
    assert STANDARD_FINGERINGS["Fork_Minor7"] == [0, 1, 1, 0, 0, 0]
    assert STANDARD_FINGERINGS["Octave_Degree_1"] == [0, 1, 1, 1, 1, 1]


def test_six_hole_reference_layout_excludes_lth():
    assert ACTIVE_HOLE_LABELS == ("L1", "L2", "L3", "R1", "R2", "R3")
    assert REFERENCE_HOLE_DIAMETERS_M == pytest.approx(
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


def test_pipeline_stage_outputs(record_property):
    pipeline = WhistleDesignPipeline("Soprano_D", IRISH_FOLK_HYBRID)
    bore = pipeline.run_stage_1_length_endhole_optimization(starting_bore_radius_m=0.0071)
    assert bore.segments[0][1] == pytest.approx(0.0071)
    assert len(bore.segments) == 3
    assert bore.segments[1][0] == pytest.approx(bore.length_m - 0.020)
    assert bore.segments[1][1] == pytest.approx(bore.segments[0][1])
    assert bore.length_m != pytest.approx(pipe_length_m(pipeline.key_config))
    assert bore.segments[-1][1] <= bore.segments[0][1]
    assert pipeline.bore_result is not None
    assert pipeline.bore_result.final_cost < pipeline.bore_result.initial_cost
    root_hz, octave_hz = pipeline.bore_result.optimized_frequencies_hz
    target_root_hz = pipeline.key_config.root_freq_12tet_hz
    assert 1200.0 * abs(math.log2(root_hz / target_root_hz)) < 10.0
    assert 1200.0 * abs(math.log2(octave_hz / (2.0 * target_root_hz))) < 10.0
    holes = pipeline.run_stage_2_tonehole_optimization(bore)
    stage_1_bore = bore
    assert pipeline.tonehole_result is not None
    bore = pipeline.tonehole_result.bore
    assert isinstance(bore, TaperedBoreGeometry)
    assert bore.length_m != pytest.approx(stage_1_bore.length_m)
    assert 0.95 * stage_1_bore.length_m <= bore.length_m <= 1.05 * stage_1_bore.length_m
    assert bore.segments[-1][0] - bore.segments[-2][0] == pytest.approx(0.020)
    assert [r for _, r in bore.segments] == [r for _, r in stage_1_bore.segments]
    assert pipeline.bore_result.bore is stage_1_bore
    assert pipeline.tonehole_result.result.success
    assert pipeline.tonehole_result.improved
    assert pipeline.tonehole_result.final_cost < pipeline.tonehole_result.initial_cost
    assert max(abs(error) for error in pipeline.tonehole_result.cent_errors.values()) < 20.0
    assert abs(pipeline.tonehole_result.cent_errors["Degree_1"]) <= 0.1001
    assert pipeline.tonehole_result.result.nfev > 0
    assert len(pipeline.tonehole_result.result.x) == 13
    assert pipeline.tonehole_result.result.x[-1] == pytest.approx(bore.length_m)
    assert bore.length_m - holes.holes[-1].position_m >= 0.030
    holes.validate(bore.length_m)
    rows = fingering_tuning_rows(
        pipeline.key_config, pipeline.temperament, bore, holes,
        pipeline.geometry_factory.load_mouthpiece("MouthpieceDN16V2"),
    )
    root = rows[0]
    assert root["Register"] == 1
    assert root["Degree"] == "Degree_1"
    assert root["Status"] == "OK"
    root_error = root["Deviation (cents)"]
    assert isinstance(root_error, float)
    assert abs(root_error) < 0.2
    record_property("stage_1_pipe_length_mm", stage_1_bore.length_m * 1000.0)
    record_property("final_pipe_length_mm", bore.length_m * 1000.0)
    record_property("closed_root_deviation_cents", root_error)
    record_property("optimized_cent_errors", pipeline.tonehole_result.cent_errors)
    drilling = drilling_chart_rows(holes, bore.length_m)
    assert drilling[-1]["Distance from end (mm)"] == pytest.approx(
        (bore.length_m - holes.holes[-1].position_m) * 1000.0
    )
    with pytest.raises(ValueError, match="tonehole_result.bore"):
        pipeline.run_stage_3_undercut_optimization(stage_1_bore, holes)
    design = pipeline.run_stage_3_undercut_optimization(bore, holes)
    assert len(design.holes.holes) == 6
    assert design.key_name == "Soprano High D"
    assert design.to_dict()["temperament"] == IRISH_FOLK_HYBRID.name
    assert design.bore is bore
    assert design.to_dict()["bore"]["segments"][-1][0] == pytest.approx(bore.length_m)


def test_key_validation():
    with pytest.raises(KeyError):
        WhistleDesignPipeline("not-a-key")
