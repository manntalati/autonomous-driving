"""Tests for utils/geometry.py — the ego-motion and TTC helpers."""
import math

import numpy as np
import pytest

from utils.geometry import (
    ego_to_global,
    global_to_ego,
    rotate_vector_ego_to_global,
    time_to_collision,
    velocity_from_track,
)


def yaw_q(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def pose(x=0.0, y=0.0, yaw=0.0):
    return {"translation": [x, y, 0.0], "rotation": yaw_q(yaw)}


class TestFrameTransforms:
    def test_roundtrip_identity(self):
        p = np.array([[10.0, 3.0], [-5.0, 2.0]])
        out = global_to_ego(ego_to_global(p, pose(100, 200, 0.7)), pose(100, 200, 0.7))
        assert np.allclose(out, p)

    def test_translation_applied_to_positions(self):
        assert np.allclose(ego_to_global(np.array([[0.0, 0.0]]), pose(5, 7)), [[5.0, 7.0]])

    def test_yaw_rotation(self):
        # 90 deg yaw sends ego +x to global +y
        out = ego_to_global(np.array([[1.0, 0.0]]), pose(0, 0, math.pi / 2))
        assert np.allclose(out, [[0.0, 1.0]], atol=1e-9)

    def test_velocity_is_rotated_not_translated(self):
        """A free vector must ignore the pose translation entirely."""
        v = np.array([[3.0, 4.0]])
        a = rotate_vector_ego_to_global(v, pose(0, 0, 0.9))
        b = rotate_vector_ego_to_global(v, pose(1000, -500, 0.9))
        assert np.allclose(a, b)
        assert np.isclose(np.linalg.norm(a), 5.0)   # rotation preserves magnitude

    def test_3d_input_preserved(self):
        p = np.array([[1.0, 2.0, 3.0]])
        assert ego_to_global(p, pose(1, 1)).shape == (1, 3)

    def test_bad_shape_rejected(self):
        with pytest.raises(ValueError):
            ego_to_global(np.zeros((2, 5)), pose())


class TestVelocityFromTrack:
    def test_parked_object_has_zero_velocity(self):
        """THE bug this guards: differencing ego-frame positions of a parked car
        under ego motion yields the ego speed, not zero."""
        # ego moves +5 m in 1 s; a globally-fixed object goes 30 -> 25 in ego frame
        v = velocity_from_track(np.array([30.0, 0.0]), pose(0, 0), 0,
                                np.array([25.0, 0.0]), pose(5, 0), 1_000_000)
        assert np.allclose(v, [0.0, 0.0], atol=1e-9)

    def test_moving_object_velocity_recovered(self):
        # object advances 10 m globally in 1 s while the ego is stationary
        v = velocity_from_track(np.array([10.0, 0.0]), pose(0, 0), 0,
                                np.array([20.0, 0.0]), pose(0, 0), 1_000_000)
        assert np.allclose(v, [10.0, 0.0], atol=1e-9)

    def test_non_positive_dt_rejected(self):
        with pytest.raises(ValueError):
            velocity_from_track(np.zeros(2), pose(), 100, np.zeros(2), pose(), 100)

    def test_global_frame_option(self):
        v = velocity_from_track(np.array([10.0, 0.0]), pose(0, 0, math.pi / 2), 0,
                                np.array([10.0, 0.0]), pose(0, 0, math.pi / 2), 1_000_000,
                                in_frame="global")
        assert np.allclose(v, [0.0, 0.0], atol=1e-9)


class TestTimeToCollision:
    def test_approaching(self):
        assert np.isclose(time_to_collision([10.0, 0.0], [-5.0, 0.0]), 2.0)

    def test_receding_is_infinite_not_negative(self):
        """An unguarded divide returns a large NEGATIVE ttc that trivially passes
        a `< threshold` test, warning about every car driving away."""
        assert time_to_collision([20.0, 0.0], [5.0, 0.0]) == float("inf")

    def test_stationary_relative_is_infinite(self):
        assert time_to_collision([20.0, 0.0], [0.0, 0.0]) == float("inf")

    def test_lateral_only_motion_does_not_collide(self):
        assert time_to_collision([20.0, 0.0], [0.0, 5.0]) == float("inf")

    def test_zero_range_is_zero(self):
        assert time_to_collision([0.0, 0.0], [-1.0, 0.0]) == 0.0
