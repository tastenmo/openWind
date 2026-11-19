from dataclasses import replace

import numpy as np
import pytest

from tin_whistle_openwind.genetic_optimizer import (
    GeneticAlgorithmConfig,
    GeneticGeometryOptimizer,
    GeometryDNA,
    _GeometryGenomeCodec,
)
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
    assert decoded_bore.segments[-2] == pytest.approx((0.23, 0.0071))
    assert decoded_bore.segments[-1] == pytest.approx((0.25, 0.0071 * 1.05))
    positions, radii = np.asarray(decoded_bore.segments).T
    assert np.interp(decoded_holes.positions_m, positions, radii) == pytest.approx(
        np.full(6, 0.0071 * 0.98)
    )


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
        config=GeneticAlgorithmConfig(include_second_register=True, mutation_scale=0.0),
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
        random_seed=42,
    )
    optimizer_args = (
        SUPPORTED_KEYS["Soprano_D"],
        IRISH_FOLK_HYBRID,
        bore,
        layout,
    )
    first = GeneticGeometryOptimizer(*optimizer_args, config=config).optimize(
        ["Degree_1", "Degree_2"]
    )
    second = GeneticGeometryOptimizer(*optimizer_args, config=config).optimize(
        ["Degree_1", "Degree_2"]
    )

    assert first.final_cost < first.initial_cost
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
