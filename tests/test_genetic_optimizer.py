from dataclasses import replace
import json

import numpy as np
import pytest

from tin_whistle_openwind.genetic_optimizer import (
    GeneticAlgorithmConfig,
    GeneticGeometryOptimizer,
    GeometryDNA,
    _GeometryGenomeCodec,
    normalized_pitch_costs,
    normalized_q_rewards,
)
from tin_whistle_openwind.genetic_winners import WinnerArchive
from tin_whistle_openwind.geometry import (
    MouthpieceGeometry,
    TaperedBoreGeometry,
    UndercutTonehole,
)
from tin_whistle_openwind.keys import SUPPORTED_KEYS
from tin_whistle_openwind.optimizer import ACTIVE_HOLE_LABELS, ToneholeLayout
from tin_whistle_openwind.tuning import IRISH_FOLK_HYBRID


def seed_geometry():
    bore = TaperedBoreGeometry(
        "seed",
        [(0.0, 0.0071), (0.23, 0.0071), (0.25, 0.0068)],
    )
    positions = [0.09, 0.12, 0.15, 0.16, 0.19, 0.219]
    holes = ToneholeLayout(
        [
            UndercutTonehole(label, position, 0.006, 0.003)
            for label, position in zip(ACTIVE_HOLE_LABELS, positions)
        ]
    )
    return bore, holes


def test_geometry_dna_vector_round_trip():
    dna = GeometryDNA(
        0.25,
        np.arange(6, dtype=float),
        np.arange(6, dtype=float) + 0.006,
        np.arange(6, dtype=float) + 0.005,
        7.5,
    )

    decoded = GeometryDNA.from_vector(dna.to_vector())

    assert decoded.pipe_length_m == pytest.approx(dna.pipe_length_m)
    assert decoded.hole_positions_m == pytest.approx(dna.hole_positions_m)
    assert decoded.bore_radii_at_holes_m == pytest.approx(dna.bore_radii_at_holes_m)
    assert decoded.hole_diameters_m == pytest.approx(dna.hole_diameters_m)
    assert decoded.bevel_deg == pytest.approx(dna.bevel_deg)
    with pytest.raises(ValueError, match="20 genes, or 21"):
        GeometryDNA.from_vector(np.zeros(19))


def test_codec_repairs_genes_and_decodes_radius_at_moving_holes():
    bore, layout = seed_geometry()
    codec = _GeometryGenomeCodec(bore, layout, GeneticAlgorithmConfig())
    genome = codec.from_geometry()
    genome[0] = 2.0
    genome[1:7] = [0.2, 0.1, 0.08, 0.07, 0.06, 0.05]
    genome[7:13] = 0.02
    genome[13:19] = 0.02
    genome[19] = 45.0

    repaired = codec.repair(genome)
    decoded_bore, decoded_holes = codec.decode(repaired)
    decoded_dna = GeometryDNA.from_vector(repaired)

    assert 0.95 * bore.length_m <= decoded_dna.pipe_length_m <= 1.05 * bore.length_m
    assert np.all(
        np.diff(decoded_holes.positions_m)
        >= np.array([0.018, 0.018, 0.0001, 0.018, 0.018]) - 1e-12
    )
    assert (
        decoded_dna.pipe_length_m - decoded_holes.positions_m[-1]
        >= 0.0301 - 1e-12
    )
    assert np.all((decoded_holes.diameters_m >= 0.005) & (decoded_holes.diameters_m <= 0.008))
    assert np.allclose(
        decoded_dna.hole_positions_m / 0.0001,
        np.round(decoded_dna.hole_positions_m / 0.0001),
    )
    assert np.allclose(
        decoded_dna.hole_diameters_m / 0.0001,
        np.round(decoded_dna.hole_diameters_m / 0.0001),
    )
    previous_radius = codec._seed_radii_at(
        decoded_dna.hole_positions_m[:1], decoded_dna.pipe_length_m
    )[0]
    for radius in decoded_dna.bore_radii_at_holes_m:
        assert 0.8 * previous_radius - 1e-12 <= radius <= previous_radius + 1e-12
        assert radius / 0.0001 == pytest.approx(round(radius / 0.0001))
        previous_radius = radius
    assert all(hole.bevel_deg == pytest.approx(15.0) for hole in decoded_holes.holes)
    bore_positions = np.array([position for position, _ in decoded_bore.segments])
    bore_radii = np.array([radius for _, radius in decoded_bore.segments])
    radii_at_holes = np.interp(decoded_holes.positions_m, bore_positions, bore_radii)
    assert radii_at_holes == pytest.approx(decoded_dna.bore_radii_at_holes_m)
    assert decoded_bore.segments[0][1] == pytest.approx(bore.segments[0][1])
    assert decoded_bore.segments[-2][0] == pytest.approx(
        decoded_dna.pipe_length_m - 0.020
    )
    assert decoded_bore.segments[-2][1] == pytest.approx(bore.segments[-2][1])
    assert decoded_bore.segments[-1][1] == pytest.approx(bore.segments[-1][1])


def test_codec_rejects_nonfinite_genes():
    bore, layout = seed_geometry()
    codec = _GeometryGenomeCodec(bore, layout, GeneticAlgorithmConfig())
    genome = codec.from_geometry()
    genome[3] = np.nan

    with pytest.raises(ValueError, match="finite"):
        codec.repair(genome)


def test_joint_geometry_genes_move_holes_bore_and_endhole_but_not_bevel():
    bore, layout = seed_geometry()
    bore = TaperedBoreGeometry.cylindrical("seed", 0.0071, 0.25)
    layout.holes[1] = replace(layout.holes[1], bevel_deg=3.0)
    codec = _GeometryGenomeCodec(
        bore, layout, GeneticAlgorithmConfig(optimize_endhole=True, optimize_bevel=False)
    )
    genome = codec.from_geometry()
    genome[1:7] += 0.0005
    genome[7:13] *= 0.98
    genome[13:19] += 0.0005
    genome[19] = 15.0
    genome[20] *= 1.2
    decoded_bore, decoded_holes = codec.decode(genome)
    assert len(genome) == 21
    assert decoded_holes.positions_m == pytest.approx(layout.positions_m + 0.0005)
    assert decoded_holes.diameters_m == pytest.approx(layout.diameters_m + 0.0005)
    assert [hole.bevel_deg for hole in decoded_holes.holes] == [0, 3, 0, 0, 0, 0]
    assert decoded_bore.segments[-2] == pytest.approx((0.23, 0.007))
    assert decoded_bore.segments[-1] == pytest.approx((0.25, 0.007))
    positions, radii = np.asarray(decoded_bore.segments).T
    hole_radii = np.interp(decoded_holes.positions_m, positions, radii)
    assert hole_radii == pytest.approx(np.full(6, 0.007))
    assert np.all(np.diff(hole_radii) <= 0.0)
    assert 0.7 * hole_radii[-1] <= decoded_bore.segments[-1][1] <= hole_radii[-1]
    assert np.all(np.diff(radii) <= 1e-12)


def test_decoded_bore_never_widens_at_seed_knots_or_terminal_taper():
    bore, layout = seed_geometry()
    bore = TaperedBoreGeometry(
        "seed",
        [(0.0, 0.0071), (0.10, 0.0071), (0.13, 0.0065),
         (0.23, 0.0071), (0.25, 0.0068)],
    )
    codec = _GeometryGenomeCodec(
        bore, layout, GeneticAlgorithmConfig(optimize_endhole=True)
    )
    genome = codec.from_geometry()
    genome[7:13] = [0.007, 0.007, 0.007, 0.007, 0.007, 0.0068]
    genome[20] = 0.0064

    repaired = GeometryDNA.from_vector(codec.repair(genome))
    decoded_bore, _ = codec.decode(genome)
    positions, radii = np.asarray(decoded_bore.segments).T

    assert np.all(np.diff(radii) <= 1e-12)
    assert np.interp(repaired.hole_positions_m, positions, radii) == pytest.approx(
        repaired.bore_radii_at_holes_m
    )
    assert decoded_bore.segments[-2] == pytest.approx((0.23, 0.0068))
    assert decoded_bore.segments[-1] == pytest.approx((0.25, 0.0064))


def test_fixed_endhole_remains_fixed_and_does_not_widen():
    bore, layout = seed_geometry()
    codec = _GeometryGenomeCodec(bore, layout, GeneticAlgorithmConfig())
    genome = codec.from_geometry()
    genome[7:13] = 0.005

    decoded_bore, _ = codec.decode(genome)
    radii = np.asarray(decoded_bore.segments)[:, 1]

    assert np.all(np.diff(radii) <= 1e-12)
    assert radii[-1] == pytest.approx(bore.segments[-1][1])
    assert np.all(radii >= bore.segments[-1][1] - 1e-12)


def test_incompatible_fixed_endhole_fails_instead_of_widening():
    bore, layout = seed_geometry()
    bore = TaperedBoreGeometry("seed", [(0.0, 0.0071), (0.25, 0.0072)])
    codec = _GeometryGenomeCodec(bore, layout, GeneticAlgorithmConfig())

    with pytest.raises(ValueError, match="contain no 0.1 mm resolution value"):
        codec.from_geometry()


def test_generation_populations_replace_duplicate_individuals(monkeypatch):
    bore, layout = seed_geometry()
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"],
        IRISH_FOLK_HYBRID,
        bore,
        layout,
        config=GeneticAlgorithmConfig(
            population_size=8,
            generations=2,
            workers=1,
            winner_directory=None,
            mutation_scale=0.0,
            optimize_endhole=True,
            optimize_bevel=False,
            random_seed=42,
        ),
    )
    monkeypatch.setattr(
        GeneticGeometryOptimizer,
        "_cent_errors",
        lambda self, vector, notes: {note: 0.0 for note in notes},
    )
    monkeypatch.setattr(
        optimizer, "_child", lambda first, second, rng: first.copy()
    )
    evaluated = []
    original_evaluate = optimizer._evaluate

    def record_evaluate(vector, notes):
        evaluated.append(tuple(vector))
        return original_evaluate(vector, notes)

    monkeypatch.setattr(optimizer, "_evaluate", record_evaluate)
    generation_sizes = []

    def record_generation(generation, cost, fitness, evaluations):
        current_generation = evaluated[-optimizer.config.population_size :]
        generation_sizes.append(len(set(current_generation)))

    optimizer.optimize(
        ["Degree_1"], progress_callback=record_generation
    )

    assert generation_sizes == [optimizer.config.population_size] * 2


def test_both_registers_score_standard_fingerings_at_their_tempered_targets(monkeypatch):
    from types import SimpleNamespace

    from tin_whistle_openwind.optimizer import OpenWInDToneholeModel
    from tin_whistle_openwind.tuning import SECOND_REGISTER_FINGERINGS, generate_key_frequencies

    bore, layout = seed_geometry()
    key = SUPPORTED_KEYS["Soprano_D"]
    calls = []

    def response(self, fingering, target, **kwargs):
        calls.append((fingering, target))
        return SimpleNamespace(peak_frequency_hz=target, admittance=np.array([0., 1., 0.]))

    monkeypatch.setattr(OpenWInDToneholeModel, "response", response)
    optimizer = GeneticGeometryOptimizer(
        key, IRISH_FOLK_HYBRID, bore, layout,
        config=GeneticAlgorithmConfig(
            include_second_register=True, mutation_scale=0.0,
            workers=1, winner_directory=None,
        ),
    )
    result = optimizer.optimize()
    assert len(result.cent_errors) == 15
    assert "Fork_Minor7" in result.cent_errors
    assert "Octave_Fork_Minor7" not in result.cent_errors
    assert result.final_cost == 0
    targets = generate_key_frequencies(key, IRISH_FOLK_HYBRID)
    fork_targets = [hz for fingering, hz in calls if fingering == "Fork_Minor7"]
    assert fork_targets and all(hz == pytest.approx(targets["Fork_Minor7"]) for hz in fork_targets)
    for label, fingering in SECOND_REGISTER_FINGERINGS.items():
        target = targets[label] if label == "Octave_Degree_1" else 2 * targets[fingering]
        assert any(name == fingering and hz == pytest.approx(target) for name, hz in calls)
        assert result.cent_errors[label] == 0


def test_root_relative_q_scoring_uses_each_register_root(monkeypatch):
    from types import SimpleNamespace

    from tin_whistle_openwind.optimizer import OpenWInDToneholeModel
    from tin_whistle_openwind.tuning import generate_key_frequencies

    bore, layout = seed_geometry()
    key = SUPPORTED_KEYS["Soprano_D"]
    targets = generate_key_frequencies(key, IRISH_FOLK_HYBRID)
    test_notes = {
        "Degree_1": (targets["Degree_1"], 10.0, 20.0),
        "Degree_2": (targets["Degree_2"], 15.0, 80.0),
        "Octave_Degree_1": (targets["Octave_Degree_1"], -8.0, 20.0),
        "Octave_Degree_2": (2 * targets["Degree_2"], -5.0, 80.0),
    }

    def response(self, fingering, target, *, f_min, f_max, n_points):
        _, error, q = min(test_notes.values(), key=lambda row: abs(row[0] - target))
        frequency = target * 2 ** (error / 1200)
        grid = np.linspace(f_min, f_max, n_points)
        amplitude = 1 / np.sqrt(1 + (2 * q * (grid - frequency) / frequency) ** 2)
        return SimpleNamespace(
            frequencies_hz=grid, impedance=1 / amplitude,
            admittance=amplitude, peak_frequency_hz=frequency,
        )

    monkeypatch.setattr(OpenWInDToneholeModel, "response", response)
    optimizer = GeneticGeometryOptimizer(
        key, IRISH_FOLK_HYBRID, bore, layout,
        config=GeneticAlgorithmConfig(
            population_size=4, generations=1, retune_closed_root=False,
            root_relative_q=True, include_second_register=True, random_seed=42,
        ),
    )
    seed = optimizer.codec.from_geometry()
    score = optimizer._score(seed, list(test_notes), retune=False)

    assert score.vector == pytest.approx(seed)
    assert score.errors == pytest.approx({
        note: values[1] for note, values in test_notes.items()
    })
    assert score.relative_errors == pytest.approx({
        "Degree_1": 10, "Degree_2": 5,
        "Octave_Degree_1": -8, "Octave_Degree_2": 3,
    })
    assert score.q_factors == pytest.approx({
        note: values[2] for note, values in test_notes.items()
    }, rel=0.01)
    expected_pitch = 0.6 * (10 * 10**2 + 5**2) / 11 + 0.4 * (10 * 8**2 + 3**2) / 11
    assert score.cost == pytest.approx(expected_pitch)
    assert score.fitness == score.cost
    import tin_whistle_openwind.genetic_optimizer as module

    original_peaks = module.resonance_peak_rows

    def unresolved_q(*args):
        return [{**row, "Q": None} for row in original_peaks(*args)]

    monkeypatch.setattr(module, "resonance_peak_rows", unresolved_q)
    without_q = optimizer._score(seed, list(test_notes), retune=False)
    assert all(q is None for q in without_q.q_factors.values())
    assert without_q.cost == pytest.approx(score.cost)
    monkeypatch.setattr(module, "resonance_peak_rows", original_peaks)
    optimizer.config = replace(optimizer.config, q_reward_weight=1.0)
    rewarded = optimizer._score(seed, list(test_notes), retune=False)
    assert rewarded.pitch_cost == pytest.approx(expected_pitch)
    assert rewarded.q_reward == pytest.approx(0.5, abs=0.005)
    assert rewarded.cost == pytest.approx(rewarded.pitch_cost - rewarded.q_reward)
    monkeypatch.setattr(module, "resonance_peak_rows", unresolved_q)
    unresolved = optimizer._score(seed, list(test_notes), retune=False)
    assert unresolved.q_reward == 0
    assert unresolved.cost == pytest.approx(expected_pitch)
    monkeypatch.setattr(module, "resonance_peak_rows", original_peaks)
    optimizer.config = replace(optimizer.config, q_reward_weight=1000)
    assert optimizer._score(seed, list(test_notes), retune=False).cost < 0


def test_normalized_register_costs_handle_different_note_counts():
    costs = normalized_pitch_costs({
        "Degree_1": 2.0, "Degree_2": 2.0, "Degree_3": 2.0,
        "Octave_Degree_1": 3.0, "Octave_Degree_2": 3.0,
    })
    assert sum(value for note, value in costs.items() if not note.startswith("Octave_")) == pytest.approx(0.6 * 4)
    assert sum(value for note, value in costs.items() if note.startswith("Octave_")) == pytest.approx(0.4 * 9)
    assert costs["Degree_1"] == pytest.approx(10 * costs["Degree_2"])
    assert costs["Octave_Degree_1"] == pytest.approx(10 * costs["Octave_Degree_2"])


@pytest.mark.parametrize("root, note", [
    ("Degree_1", "Degree_2"),
    ("Octave_Degree_1", "Octave_Degree_2"),
])
def test_single_register_cost_is_renormalized(root, note):
    costs = normalized_pitch_costs({root: 2.0, note: 3.0})
    assert sum(costs.values()) == pytest.approx((10 * 4 + 9) / 11)


def test_q_rewards_are_bounded_and_normalized_without_extra_root_weight():
    q = {
        "Degree_1": 40.0, "Degree_2": 40.0, "Degree_3": None,
        "Octave_Degree_1": 80.0,
    }
    rewards = normalized_q_rewards(q)
    assert rewards["Degree_1"] == pytest.approx(rewards["Degree_2"])
    assert rewards["Degree_3"] == 0
    assert sum(rewards.values()) == pytest.approx(0.6 / 3 + 0.4 * 2 / 3)
    assert sum(normalized_q_rewards(q, weight=0).values()) == 0
    single = normalized_q_rewards({"Octave_Degree_1": 40.0})
    assert sum(single.values()) == pytest.approx(0.5)
    higher = normalized_q_rewards({note: 1e6 for note in q})
    assert sum(rewards.values()) < sum(higher.values()) < 1
    assert higher["Degree_1"] > rewards["Degree_1"]


@pytest.mark.parametrize("changes", [
    {"max_outer_diameter_m": float("nan")},
    {"minimum_wall_m": 0.008},
    {"q_reward_scale": 0},
    {"q_reward_weight": -1},
    {"q_reward_weight": 1},
])
def test_invalid_manufacturing_or_q_settings_rejected(changes):
    with pytest.raises(ValueError):
        GeneticAlgorithmConfig(**changes)


@pytest.mark.parametrize("endhole", [None, 0.005])
def test_independent_hole_dna_round_trip(endhole):
    dna = GeometryDNA(
        0.25, np.linspace(0.08, 0.21, 6), np.full(6, 0.006),
        np.full(6, 0.006), 0.0, endhole,
        chimney_heights_m=np.linspace(0.0009, 0.0014, 6),
        hole_bevels_deg=np.arange(6, dtype=float),
    )
    vector = dna.to_vector()
    decoded = GeometryDNA.from_vector(vector)
    assert decoded.to_vector() == pytest.approx(vector)
    assert decoded.chimney_heights_m == pytest.approx(dna.chimney_heights_m)
    assert decoded.hole_bevels_deg == pytest.approx(dna.hole_bevels_deg)
    assert decoded.endhole_radius_m == endhole


def test_independent_holes_and_offspring_obey_coupled_constraints():
    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(
        independent_hole_geometry=True, optimize_endhole=True,
        mutation_rate=1.0, mutation_scale=1.0, random_seed=11,
        winner_directory=None, workers=1,
    )
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout, config=config,
    )
    codec = optimizer.codec
    seed = GeometryDNA.from_vector(codec.from_geometry())
    assert seed.chimney_heights_m == pytest.approx(np.full(6, 0.0009))
    seed.bore_radii_at_holes_m = np.linspace(0.006, 0.0055, 6)
    seed.chimney_heights_m = np.linspace(0.0009, 0.0014, 6)
    seed.hole_bevels_deg = np.arange(1, 7, dtype=float)
    first = codec.repair(seed.to_vector())
    _, independent = codec.decode(first)
    assert [h.chimney_height_m for h in independent.holes] == pytest.approx(seed.chimney_heights_m)
    assert [h.bevel_deg for h in independent.holes] == pytest.approx(seed.hole_bevels_deg)
    extreme = GeometryDNA.from_vector(first)
    extreme.bore_radii_at_holes_m[:] = 0.001
    extreme.hole_diameters_m[:] = 0.008
    extreme.chimney_heights_m[:] = 0.1
    extreme.hole_bevels_deg[:] = 90
    second = codec.repair(extreme.to_vector())
    rng = np.random.default_rng(42)
    for vector in [first, second, *[
        optimizer._child(first, second, rng) for _ in range(40)
    ]]:
        dna = GeometryDNA.from_vector(vector)
        decoded_bore, decoded_holes = codec.decode(vector)
        assert codec.repair(vector) == pytest.approx(vector, abs=1e-14)
        assert dna.dimensional_values / 0.0001 == pytest.approx(
            np.round(dna.dimensional_values / 0.0001)
        )
        assert np.all(np.diff(np.asarray(decoded_bore.segments)[:, 1]) <= 1e-12)
        for hole, radius in zip(decoded_holes.holes, dna.bore_radii_at_holes_m):
            assert 0.0009 - 1e-12 <= hole.chimney_height_m
            assert 2 * (radius + hole.chimney_height_m) <= 0.016 + 1e-12
            assert hole.bevel_side == "bore"
            assert 0 <= hole.bevel_deg <= 15
            assert hole.radius_out_m <= hole.radius_in_m <= radius + 1e-12
            assert 0.005 - 1e-12 <= 2 * hole.radius_out_m <= 0.008 + 1e-12


def test_incompatible_fixed_body_envelope_fails_explicitly():
    bore, layout = seed_geometry()
    with pytest.raises(ValueError, match="outer diameter/minimum wall"):
        _GeometryGenomeCodec(bore, layout, GeneticAlgorithmConfig(
            independent_hole_geometry=True, max_outer_diameter_m=0.0159,
        ))


def test_root_relative_q_penalizes_missing_resonances(monkeypatch):
    from types import SimpleNamespace

    from tin_whistle_openwind.optimizer import OpenWInDToneholeModel

    bore, layout = seed_geometry()
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
        config=GeneticAlgorithmConfig(
            retune_closed_root=False, root_relative_q=True,
        ),
    )
    monkeypatch.setattr(
        OpenWInDToneholeModel, "response",
        lambda self, fingering, target, **kwargs: SimpleNamespace(
            admittance=np.array([1.0, 2.0, 3.0]), peak_frequency_hz=target
        ),
    )

    score = optimizer._score(optimizer.codec.from_geometry(), ["Degree_2"], retune=False)

    assert score.relative_errors["Degree_2"] == 1200.0
    assert score.q_factors["Degree_2"] is None
    assert score.cost == 1200.0**2


def test_root_relative_q_requires_retuning_disabled():
    with pytest.raises(ValueError, match="requires retune_closed_root=False"):
        GeneticAlgorithmConfig(root_relative_q=True)


def test_root_relative_q_scoring_runs_against_openwind():
    bore, layout = seed_geometry()
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
        mouthpiece=MouthpieceGeometry.load("MouthpieceDN16V2"),
        config=GeneticAlgorithmConfig(
            population_size=4, generations=1, random_seed=3,
            workers=1, winner_directory=None,
            retune_closed_root=False, root_relative_q=True,
            include_second_register=True,
        ),
    )
    notes = ["Degree_1", "Degree_2", "Octave_Degree_1", "Octave_Degree_2"]

    result = optimizer.optimize(notes)

    assert np.isfinite(result.final_cost)
    assert set(result.cent_errors) == set(notes)
    assert set(result.relative_cent_errors) == set(notes)
    assert set(result.q_factors) == set(notes)
    assert all(q is None or q > 0 for q in result.q_factors.values())


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_rejected(workers):
    with pytest.raises(ValueError, match="workers"):
        GeneticAlgorithmConfig(workers=workers)


def test_worker_count_respects_auto_and_population(monkeypatch):
    import tin_whistle_openwind.genetic_optimizer as module

    bore, layout = seed_geometry()
    if hasattr(module.os, "process_cpu_count"):
        monkeypatch.setattr(module.os, "process_cpu_count", lambda: 8)
    else:
        monkeypatch.setattr(module.os, "cpu_count", lambda: 8)
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
        config=GeneticAlgorithmConfig(population_size=4, winner_directory=None),
    )
    assert optimizer._worker_count() == 4
    optimizer.config = replace(optimizer.config, workers=1)
    assert optimizer._worker_count() == 1


def test_parallel_and_serial_openwind_results_match():
    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(
        population_size=4, generations=1, random_seed=11,
        retune_closed_root=False, root_relative_q=True, include_second_register=True,
        winner_directory=None, workers=1,
        independent_hole_geometry=True, optimize_endhole=True,
        q_reward_weight=1.0,
    )
    notes = ["Degree_1", "Degree_2", "Octave_Degree_1", "Octave_Degree_2"]
    reports = []
    results = []
    for workers in (1, 2):
        run_reports = []
        optimizer = GeneticGeometryOptimizer(
            SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
            mouthpiece=MouthpieceGeometry.load("MouthpieceDN16V2"),
            config=replace(config, workers=workers),
        )
        results.append(optimizer.optimize(notes, progress_callback=lambda *args: run_reports.append(args)))
        reports.append(run_reports)

    serial, parallel = results
    assert parallel.dna.to_vector() == pytest.approx(serial.dna.to_vector())
    assert parallel.final_cost == pytest.approx(serial.final_cost)
    assert parallel.cent_errors == pytest.approx(serial.cent_errors)
    assert parallel.relative_cent_errors == pytest.approx(serial.relative_cent_errors)
    assert parallel.q_factors == pytest.approx(serial.q_factors)
    assert parallel.evaluations == serial.evaluations
    assert reports[1] == pytest.approx(reports[0])
    assert parallel.final_cost == pytest.approx(parallel.final_pitch_cost - parallel.final_q_reward)
    assert 0 < parallel.final_q_reward < 1
    assert all(hole.bevel_side == "bore" for hole in parallel.layout.holes)


def test_worker_errors_propagate_and_executor_closes():
    from concurrent.futures import ProcessPoolExecutor

    from tin_whistle_openwind.genetic_optimizer import _initialize_worker

    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(workers=2, winner_directory=None)
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout, config=config,
    )
    with ProcessPoolExecutor(
        max_workers=2, initializer=_initialize_worker,
        initargs=(optimizer.key_config, optimizer.tuning, bore, layout, 20.0, None, config),
    ) as executor:
        with pytest.raises(KeyError):
            optimizer._evaluate_population([optimizer.codec.from_geometry()], ("Invalid",), executor)
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(abs, -1)


def test_parallel_cache_scores_duplicate_genomes_once():
    from concurrent.futures import ProcessPoolExecutor

    from tin_whistle_openwind.genetic_optimizer import _initialize_worker

    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(
        workers=2, winner_directory=None, retune_closed_root=False, root_relative_q=True,
    )
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout, config=config,
    )
    genome = optimizer.codec.from_geometry()
    with ProcessPoolExecutor(
        max_workers=2, initializer=_initialize_worker,
        initargs=(optimizer.key_config, optimizer.tuning, bore, layout, 20.0, None, config),
    ) as executor:
        scores = optimizer._evaluate_population([genome, genome.copy()], ("Degree_1",), executor)
        again = optimizer._evaluate_population([genome], ("Degree_1",), executor)
    assert optimizer._evaluations == 1
    assert scores[0] is scores[1] is again[0]


def test_parallel_warm_start_matches_serial(tmp_path):
    from shutil import copytree

    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(
        population_size=4, generations=1, random_seed=42, workers=1,
        retune_closed_root=False, root_relative_q=True,
        winner_directory=tmp_path / "seed",
    )
    args = (SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout)
    GeneticGeometryOptimizer(*args, config=config).optimize(["Degree_1"])
    copytree(tmp_path / "seed", tmp_path / "serial")
    copytree(tmp_path / "seed", tmp_path / "parallel")

    serial = GeneticGeometryOptimizer(
        *args, config=replace(config, winner_directory=tmp_path / "serial"),
    ).optimize(["Degree_1"])
    parallel = GeneticGeometryOptimizer(
        *args, config=replace(config, workers=2, winner_directory=tmp_path / "parallel"),
    ).optimize(["Degree_1"])

    assert parallel.dna.to_vector() == pytest.approx(serial.dna.to_vector())
    assert parallel.final_cost == pytest.approx(serial.final_cost)
    assert parallel.evaluations == serial.evaluations


def test_winner_archive_reuses_and_rescores_compatible_candidates(tmp_path, monkeypatch):
    bore, layout = seed_geometry()
    scored = []

    def fake_errors(self, vector, notes):
        scored.append(tuple(vector))
        return {note: float((vector[13] * 1000 - 6.3) * 10) for note in notes}

    monkeypatch.setattr(GeneticGeometryOptimizer, "_cent_errors", fake_errors)
    config = GeneticAlgorithmConfig(
        population_size=4, generations=2, random_seed=11, workers=1,
        retune_closed_root=False, winner_directory=tmp_path,
    )
    args = (SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout)
    first = GeneticGeometryOptimizer(*args, config=config)
    first.optimize(["Degree_2"])
    archive = first._winner_archive(("Degree_2",))
    saved = archive.load()
    assert 1 < len(saved) <= 10
    assert len(set(saved)) == len(saved)

    second = GeneticGeometryOptimizer(*args, config=replace(config, random_seed=12))
    assert second._winner_archive(("Degree_2",)).path == archive.path
    assert second._winner_archive(("Degree_3",)).path != archive.path
    changed_score = GeneticGeometryOptimizer(
        *args, config=replace(config, root_relative_q=True)
    )
    assert changed_score._winner_archive(("Degree_2",)).path != archive.path
    changed_bore = TaperedBoreGeometry("seed", [
        (0.0, 0.0072), (0.23, 0.0072), (0.25, 0.0068),
    ])
    changed_geometry = GeneticGeometryOptimizer(
        args[0], args[1], changed_bore, layout, config=config,
    )
    assert changed_geometry._winner_archive(("Degree_2",)).path != archive.path
    captured = []
    initial_population = []
    original = second._initial_population

    def capture(seed, rng, winners=()):
        captured.extend(tuple(winner) for winner in winners)
        population = original(seed, rng, winners)
        initial_population.extend(tuple(candidate) for candidate in population)
        return population

    monkeypatch.setattr(second, "_initial_population", capture)
    scored.clear()
    second.optimize(["Degree_2"])
    assert captured
    assert all(tuple(second.codec.repair(np.asarray(candidate))) in scored for candidate in saved)
    assert initial_population[0] == tuple(second.codec.from_geometry())
    assert len(set(initial_population)) == config.population_size
    assert any(candidate in captured for candidate in initial_population[1:])
    assert len(archive.load()) <= 10


def test_winner_archive_rejects_corruption_and_disable_avoids_files(tmp_path, monkeypatch):
    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(workers=1, winner_directory=tmp_path)
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout, config=config,
    )
    archive = optimizer._winner_archive(("Degree_1",))
    archive.path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid genetic winner archive"):
        archive.load()
    disabled = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
        config=replace(config, winner_directory=None),
    )
    assert disabled._winner_archive(("Degree_1",)) is None


def test_winner_archive_concurrent_writes_merge(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    archive = WinnerArchive(tmp_path, {"seed": "same"})
    entries = [
        ((float(index),) * 20, (0.0, float(index)))
        for index in range(20)
    ]
    with ProcessPoolExecutor(max_workers=2) as executor:
        list(executor.map(archive.merge, (entries[:10], entries[10:])))

    assert archive.load() == [entry[0] for entry in entries[:10]]
    payload = json.loads(archive.path.read_text(encoding="utf-8"))
    assert len(payload["winners"]) == 10


def test_new_winner_schema_preserves_negative_costs_and_context(tmp_path):
    bore, layout = seed_geometry()
    config = GeneticAlgorithmConfig(
        independent_hole_geometry=True, optimize_endhole=True,
        root_relative_q=True, retune_closed_root=False, q_reward_weight=1,
        winner_directory=tmp_path,
    )
    args = SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout
    optimizer = GeneticGeometryOptimizer(*args, config=config)
    archive = optimizer._winner_archive(("Degree_1",))
    seed = optimizer.codec.from_geometry()
    other = seed.copy()
    other[1] += 0.0001
    archive.merge([(tuple(seed), (0, -0.5)), (tuple(other), (0, -0.75))])
    assert archive.load() == [tuple(other), tuple(seed)]
    for changes in (
        {"q_reward_weight": 0}, {"q_reward_scale": 50},
        {"max_outer_diameter_m": 0.017}, {"minimum_wall_m": 0.0008},
        {"independent_hole_geometry": False},
    ):
        changed = GeneticGeometryOptimizer(*args, config=replace(config, **changes))
        assert changed._winner_archive(("Degree_1",)).path != archive.path
    payload = json.loads(archive.path.read_text())
    payload["winners"][0]["rank"][0] = -1
    archive.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid genetic winner"):
        archive.load()


def test_genetic_search_is_seeded_and_refines_fitness(monkeypatch):
    bore, layout = seed_geometry()

    def fake_cent_errors(self, vector, notes):
        dna = GeometryDNA.from_vector(self.codec.repair(vector))
        return {
            note: 0.0 if note == "Degree_1" else dna.bevel_deg - 8.0
            for note in notes
        }

    monkeypatch.setattr(GeneticGeometryOptimizer, "_cent_errors", fake_cent_errors)
    config = GeneticAlgorithmConfig(
        population_size=12,
        generations=12,
        workers=1,
        winner_directory=None,
        random_seed=42,
    )
    optimizer_args = (
        SUPPORTED_KEYS["Soprano_D"],
        IRISH_FOLK_HYBRID,
        bore,
        layout,
    )
    generation_reports = []

    def record_generation(generation, cost, fitness, evaluations):
        generation_reports.append((generation, cost, fitness, evaluations))

    first = GeneticGeometryOptimizer(*optimizer_args, config=config).optimize(
        ["Degree_1", "Degree_2"],
        progress_callback=record_generation,
    )
    second = GeneticGeometryOptimizer(*optimizer_args, config=config).optimize(
        ["Degree_1", "Degree_2"]
    )

    assert first.final_cost < first.initial_cost
    assert [report[0] for report in generation_reports] == list(
        range(1, config.generations + 1)
    )
    assert all(report[3] > 0 for report in generation_reports)
    assert first.final_fitness == pytest.approx(second.final_fitness)
    assert first.dna.to_vector() == pytest.approx(second.dna.to_vector())
    assert first.root_tuning_satisfied is True
    first.layout.validate(first.bore.length_m)


def test_genetic_search_runs_against_openwind():
    bore, layout = seed_geometry()
    layout = ToneholeLayout([
        replace(hole, position_m=hole.position_m - 0.012) for hole in layout.holes
    ])
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"],
        IRISH_FOLK_HYBRID,
        bore,
        layout,
        mouthpiece=MouthpieceGeometry.load("MouthpieceDN16V2"),
        config=GeneticAlgorithmConfig(
            population_size=4,
            generations=1,
            workers=1,
            winner_directory=None,
            random_seed=3,
        ),
    )

    result = optimizer.optimize(["Degree_1"])

    assert result.evaluations > 0
    assert result.bore.length_m == pytest.approx(result.dna.pipe_length_m)
    assert result.root_tuning_satisfied
    assert abs(result.cent_errors["Degree_1"]) <= 0.1
    result.layout.validate(result.bore.length_m)


@pytest.mark.parametrize("retune, expected_bevel", [(True, 8.0), (False, 0.0)])
def test_length_retuning_allows_an_octave_improvement(monkeypatch, retune, expected_bevel):
    bore, layout = seed_geometry()
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
        config=GeneticAlgorithmConfig(
            population_size=4, generations=1, mutation_rate=0.0,
            workers=1, winner_directory=None,
            crossover_rate=0.0, retune_closed_root=retune, random_seed=42,
        ),
    )

    def errors(vector, notes):
        length, bevel = vector[0], vector[-1]
        values = {
            "Degree_1": 6000 * (0.25 + 0.0005 * bevel - length),
            "Octave_Degree_1": 16 - 2 * bevel,
        }
        return {note: values[note] for note in notes}

    def population(seed, rng):
        candidate = seed.copy()
        candidate[-1] = 8.0
        return np.array([seed, candidate, candidate, candidate])

    monkeypatch.setattr(optimizer, "_cent_errors", errors)
    monkeypatch.setattr(optimizer, "_initial_population", population)
    result = optimizer.optimize(["Degree_1", "Octave_Degree_1"])

    assert result.initial_cent_errors["Octave_Degree_1"] == 16.0
    assert result.dna.bevel_deg == expected_bevel
    assert result.bore.length_m == pytest.approx(0.25 + expected_bevel * 0.0005)
    assert result.cent_errors["Octave_Degree_1"] == 16 - 2 * expected_bevel
    assert result.root_tuning_satisfied is True
    assert result.layout.positions_m == pytest.approx(layout.positions_m)


def test_unreachable_root_reports_closest_bounded_trial(monkeypatch):
    bore, layout = seed_geometry()
    optimizer = GeneticGeometryOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout,
        config=GeneticAlgorithmConfig(
            population_size=4, generations=1, mutation_scale=0.0, random_seed=42,
            workers=1, winner_directory=None,
        ),
    )
    monkeypatch.setattr(
        optimizer, "_cent_errors",
        lambda vector, notes: {note: 6000 * (0.3 - vector[0]) for note in notes},
    )
    result = optimizer.optimize(["Degree_1"])
    assert result.bore.length_m == pytest.approx(1.05 * bore.length_m)
    assert result.root_tuning_satisfied is False
    assert np.isinf(result.final_fitness)
    result.layout.validate(result.bore.length_m)
