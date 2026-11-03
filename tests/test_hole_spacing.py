import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tin_whistle_openwind.geometry import TaperedBoreGeometry, UndercutTonehole
from tin_whistle_openwind.keys import SUPPORTED_KEYS
from tin_whistle_openwind.optimizer import (
    ACTIVE_HOLE_LABELS,
    InverseToneholeOptimizer,
    ToneholeLayout,
    ToneholeOptimizer,
)
from tin_whistle_openwind.tuning import IRISH_FOLK_HYBRID


def layout_with_positions(positions):
    return ToneholeLayout([
        UndercutTonehole(label, position, 0.006, 0.003)
        for label, position in zip(ACTIVE_HOLE_LABELS, positions)
    ])


def test_short_gap_between_hands_is_valid():
    layout_with_positions([0.1, 0.12, 0.14, 0.15, 0.17, 0.19]).validate(0.25)


@pytest.mark.parametrize("index", [0, 1, 3, 4])
def test_within_hand_minimum_is_still_enforced(index):
    positions = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 0.20])
    positions[index + 1] = positions[index] + 0.017
    with pytest.raises(ValueError, match="within each hand"):
        layout_with_positions(positions).validate(0.25)


@pytest.mark.parametrize("r1", [0.14, 0.139])
def test_hand_gap_still_requires_order(r1):
    with pytest.raises(ValueError, match="ordered"):
        layout_with_positions([0.1, 0.12, 0.14, r1, 0.17, 0.19]).validate(0.25)


def test_solver_constraints_allow_short_hand_gap(monkeypatch):
    layout = layout_with_positions([0.1, 0.12, 0.14, 0.15, 0.17, 0.19])
    optimizer = ToneholeOptimizer(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID,
        TaperedBoreGeometry.cylindrical("test", 0.0071, 0.25),
    )
    monkeypatch.setattr(optimizer, "objective", lambda vector, notes: 0.0)
    monkeypatch.setattr(optimizer, "_cent_error", lambda model, note: 0.0)

    def minimize(objective, x0, *, bounds, constraints, **kwargs):
        assert all(constraint["fun"](x0) >= 0 for constraint in constraints)
        assert constraints[2]["fun"](x0) == pytest.approx(9.9)
        for index in (0, 1, 3, 4):
            probe = x0.copy()
            probe[index + 1] = probe[index] + 17
            assert constraints[index]["fun"](probe) < 0
        probe = x0.copy()
        probe[3] = probe[2] - 1
        assert constraints[2]["fun"](probe) < 0
        assert bounds[3][0] == pytest.approx(54)
        return OptimizeResult(x=x0, fun=0.0, success=True, jac=None)

    monkeypatch.setattr("tin_whistle_openwind.optimizer.minimize", minimize)
    result = optimizer.optimize(layout.positions_m, layout.diameters_m)
    assert result.layout.positions_m == pytest.approx(layout.positions_m)


def test_inverse_uses_only_ordering_for_hand_gap(monkeypatch):
    constraints = []

    class Geometry:
        def __init__(self, *args):
            pass

        def constrain_2_holes_distance(self, first, second, *, Lmin):
            constraints.append((first, second, Lmin))

    class Inverse:
        def __init__(self, *args, **kwargs):
            pass

        def get_cost_grad_hessian(self):
            return (0.0,)

        def optimize_freq_model(self, **kwargs):
            return OptimizeResult(success=True)

    monkeypatch.setattr("tin_whistle_openwind.optimizer.InstrumentGeometry", Geometry)
    monkeypatch.setattr("tin_whistle_openwind.optimizer.InstrumentPhysics", lambda *a, **k: None)
    monkeypatch.setattr("tin_whistle_openwind.optimizer.InverseFrequentialResponse", Inverse)
    optimizer = InverseToneholeOptimizer(
        TaperedBoreGeometry.cylindrical("test", 0.0071, 0.25),
        layout_with_positions([0.1, 0.12, 0.14, 0.15, 0.17, 0.19]),
    )
    optimizer.fit(np.array([500, 600]), [np.ones(2)], ["Degree_1"])
    assert constraints == [
        ("L1", "L2", 0.018), ("L2", "L3", 0.018),
        ("L3", "R1", 0.0001), ("R1", "R2", 0.018), ("R2", "R3", 0.018),
    ]
