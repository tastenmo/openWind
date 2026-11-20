from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

import tin_whistle_openwind.pipeline as pipeline_module
from tin_whistle_openwind.geometry import TaperedBoreGeometry, UndercutTonehole
from tin_whistle_openwind.keys import (
    SUPPORTED_KEYS,
    inner_radius_m,
    pipe_length_m,
    scale_mouthpiece_length_m,
)
from tin_whistle_openwind.pipeline import WhistleDesignPipeline
from tin_whistle_openwind.optimize_bore import optimize_length_endhole
from tin_whistle_openwind.optimizer import (
    ACTIVE_HOLE_LABELS,
    REFERENCE_HOLE_DIAMETERS_M,
    ToneholeLayout,
)
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


def test_length_endhole_optimizer_can_use_straight_pvc_bore():
    radius_m = 0.0071
    result = optimize_length_endhole(
        SUPPORTED_KEYS["Soprano_D"],
        starting_bore_radius_m=radius_m,
        optimize_endhole=False,
    )

    assert len(result.bore.segments) == 2
    assert [radius for _, radius in result.bore.segments] == pytest.approx([radius_m, radius_m])
    assert result.final_cost < result.initial_cost


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
    maximum_bevel = UndercutTonehole("maximum", 0.1, 0.006, 0.003, bevel_deg=30.0)
    assert maximum_bevel.bevel_deg == 30.0
    with pytest.raises(ValueError):
        UndercutTonehole("bad", 0.1, 0.006, 0.003, bevel_deg=31.0)


def test_conical_bore_side_hole_survives_acoustic_offset_and_export(monkeypatch):
    from dataclasses import replace
    import numpy as np
    import tin_whistle_openwind.optimizer as optimizer_module
    from tin_whistle_openwind.optimize_undercut import optimize_undercut

    hole = UndercutTonehole("L1", 0.09, 0.006, 0.0012, 12.0, bevel_side="bore")
    assert hole.radius_out_m == pytest.approx(0.003)
    assert hole.radius_in_m == pytest.approx(0.003 + 0.0012 * math.tan(math.radians(12)))
    assert hole.to_row()[3:] == pytest.approx([
        hole.radius_in_m, hole.radius_out_m, hole.chimney_height_m,
    ])
    layout = ToneholeLayout([hole])
    copied = optimize_undercut(layout, bevel_deg=5).layout.holes[0]
    assert copied == replace(hole, bevel_deg=5)
    captured = []

    def computation(frequencies, main_bore, holes, *args, **kwargs):
        captured.extend(holes)
        return SimpleNamespace(impedance=np.ones(len(frequencies), dtype=complex))

    monkeypatch.setattr(optimizer_module, "ImpedanceComputation", computation)
    bore = TaperedBoreGeometry.cylindrical("test", 0.0071, 0.25)
    model = optimizer_module.OpenWInDToneholeModel(bore, layout, hole_position_offset_m=0.035)
    model.impedance(np.array([500., 600., 700.]), "Degree_1")
    assert captured[1][2:] == pytest.approx([
        0.125, hole.radius_in_m, hole.radius_out_m, hole.chimney_height_m,
    ])
    pipeline = WhistleDesignPipeline("Soprano_D")
    pipeline.genetic_result = SimpleNamespace(
        bore=bore, layout=layout, dna=SimpleNamespace(endhole_radius_m=None),
    )
    design = pipeline.run_stage_3_undercut_optimization(bore, layout)
    assert design.holes is layout
    saved_hole = design.to_dict()["holes"][0]
    assert saved_hole["bevel_side"] == "bore"
    assert saved_hole["radius_in_m"] == hole.radius_in_m
    assert saved_hole["radius_out_m"] == hole.radius_out_m


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


def test_pipeline_uses_requested_mouthpiece_through_all_stages(monkeypatch):
    pipeline = WhistleDesignPipeline(
        "Soprano_D",
        mouthpiece_name="MouthpieceDN16V4",
        optimize_endhole=False,
    )
    bore = TaperedBoreGeometry.cylindrical("test bore", 0.0071, 0.24)
    layout = ToneholeLayout([])
    optimized_mouthpieces = []

    assert pipeline.mouthpiece.name == "MouthpieceDN16V4"
    assert pipeline.mouthpiece.segments[-1][0] == pytest.approx(0.035)

    def fake_optimize_length_endhole(
        key_config, mouthpiece, starting_bore_radius_m=None, optimize_endhole=True
    ):
        optimized_mouthpieces.append((mouthpiece, optimize_endhole))
        return SimpleNamespace(bore=bore)

    def fake_optimize_toneholes(key_config, optimized_bore, temperament, mouthpiece=None):
        optimized_mouthpieces.append(mouthpiece)
        return SimpleNamespace(layout=layout, bore=bore)

    def fake_optimize_genetically(
        key_config, optimized_bore, optimized_layout, temperament, **kwargs
    ):
        return SimpleNamespace(
            bore=optimized_bore,
            layout=optimized_layout,
            dna=SimpleNamespace(endhole_radius_m=0.0071),
        )

    bevels = []
    monkeypatch.setattr(pipeline_module, "optimize_length_endhole", fake_optimize_length_endhole)
    monkeypatch.setattr(pipeline_module, "optimize_toneholes", fake_optimize_toneholes)
    monkeypatch.setattr(
        pipeline_module, "optimize_geometry_genetically", fake_optimize_genetically
    )
    monkeypatch.setattr(
        pipeline_module,
        "optimize_undercut",
        lambda optimized_layout, bevel_deg=0.0: (
            bevels.append(bevel_deg) or SimpleNamespace(layout=optimized_layout)
        ),
    )

    pipeline.run_stage_1_length_endhole_optimization()
    pipeline.run_stage_2_tonehole_optimization(bore)
    pipeline.run_genetic_refinement(bore, layout)
    design = pipeline.run_stage_3_undercut_optimization(bore, layout)

    assert optimized_mouthpieces == [
        (pipeline.mouthpiece, False),
        pipeline.mouthpiece,
    ]
    assert bevels == []
    assert design.holes is layout
    assert design.metadata["mouthpiece_name"] == "MouthpieceDN16V4"
    assert design.metadata["endhole_optimized"] is True
