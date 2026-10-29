"""Unit tests for the angle maths."""
import math
from types import SimpleNamespace

import kinematics


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def test_straight_joint_is_pi():
    assert math.isclose(kinematics.calculate_angle(point(0, 0), point(1, 0), point(2, 0)), math.pi)


def test_right_angle_is_half_pi():
    assert math.isclose(kinematics.calculate_angle(point(0, 1), point(0, 0), point(1, 0)),
                        math.pi / 2)


def test_parallel_fingers_have_zero_spread():
    spread = kinematics.calculate_spread(point(0, 0), point(0, 1), point(1, 0), point(1, 1))
    assert math.isclose(spread, 0.0, abs_tol=1e-6)


def test_perpendicular_fingers_have_half_pi_spread():
    spread = kinematics.calculate_spread(point(0, 0), point(0, 1), point(0, 0), point(1, 0))
    assert math.isclose(spread, math.pi / 2)


def test_get_leap_state_returns_16_finite_angles():
    # Synthetic hand: 21 distinct, non-degenerate landmarks
    landmarks = [point(0.05 * i, 0.02 * i + 0.03 * (i % 3), 0.004 * i + 0.01 * (i % 2))
                 for i in range(21)]
    state = kinematics.get_leap_state(SimpleNamespace(landmark=landmarks))

    assert set(state) == {"index", "middle", "ring", "thumb"}
    assert set(state["thumb"]) == {"abd", "base", "pip", "dip"}
    for finger in ("index", "middle", "ring"):
        assert set(state[finger]) == {"abd", "mcp", "pip", "dip"}
    values = [v for joints in state.values() for v in joints.values()]
    assert len(values) == 16
    assert all(math.isfinite(v) for v in values)
