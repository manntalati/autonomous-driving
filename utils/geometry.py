"""
Ego-frame / global-frame geometry helpers.

WHY THIS EXISTS (blocker fix)
-----------------------------
Detections are produced in the EGO frame, which is bolted to a moving vehicle.
A parked car has a *changing* ego-frame position purely because the ego vehicle
is driving past it. Differencing ego-frame positions across frames therefore
measures ego motion, not object motion: every stationary object appears to close
on you at exactly the ego speed, and a time-to-collision rule built on it fires a
collision warning for every parked car on the street.

The fix is to lift positions into the global (map) frame using each frame's
`ego_pose` before differencing, then convert the resulting velocity back into the
current ego frame if you want it expressed relative to the vehicle.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from pyquaternion import Quaternion


def ego_pose_matrix(ego_pose: dict) -> np.ndarray:
    """
    Build the 4x4 ego->global homogeneous transform from a nuScenes ego_pose record.

    Args: ego_pose — record with "translation" (3,) and "rotation" (quaternion wxyz).
    Returns: (4, 4) float64 matrix.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
    m[:3, 3] = np.asarray(ego_pose["translation"], dtype=np.float64)
    return m


def ego_to_global(points_ego: np.ndarray, ego_pose: dict) -> np.ndarray:
    """
    Args: points_ego — (N, 2) or (N, 3) ego-frame positions; ego_pose — record.
    Returns: same shape, in the global frame.

    2-D input is treated as z=0, which is correct for BEV centres.
    """
    pts = np.asarray(points_ego, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"expected (N,2) or (N,3), got {pts.shape}")
    two_d = pts.shape[1] == 2
    if two_d:
        pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    m = ego_pose_matrix(ego_pose)
    out = pts @ m[:3, :3].T + m[:3, 3]
    return out[:, :2] if two_d else out


def global_to_ego(points_global: np.ndarray, ego_pose: dict) -> np.ndarray:
    """Inverse of ego_to_global."""
    pts = np.asarray(points_global, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"expected (N,2) or (N,3), got {pts.shape}")
    two_d = pts.shape[1] == 2
    if two_d:
        pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    m = ego_pose_matrix(ego_pose)
    out = (pts - m[:3, 3]) @ m[:3, :3]          # R^T (x - t), since R is orthonormal
    return out[:, :2] if two_d else out


def rotate_vector_ego_to_global(vectors_ego: np.ndarray, ego_pose: dict) -> np.ndarray:
    """
    Rotate a VELOCITY (or any free vector) from ego to global.

    Velocities are directions, not positions: rotate them, never translate them.
    Adding the ego translation to a velocity is the classic sensor-fusion bug —
    it produces plausible-looking numbers that are wrong by the ego position.
    """
    v = np.asarray(vectors_ego, dtype=np.float64)
    two_d = v.shape[-1] == 2
    if two_d:
        v = np.concatenate([v, np.zeros((*v.shape[:-1], 1))], axis=-1)
    R = Quaternion(ego_pose["rotation"]).rotation_matrix
    out = v @ R.T
    return out[..., :2] if two_d else out


def velocity_from_track(
    p0_ego: np.ndarray,
    pose0: dict,
    t0_us: int,
    p1_ego: np.ndarray,
    pose1: dict,
    t1_us: int,
    in_frame: str = "ego",
    ref_pose: Optional[dict] = None,
) -> np.ndarray:
    """
    Ego-motion-compensated velocity of a tracked object between two frames.

    Args:
        p0_ego, p1_ego: (2,) or (N, 2) ego-frame positions at the two times.
        pose0, pose1: the ego_pose records for those two times.
        t0_us, t1_us: nuScenes microsecond timestamps.
        in_frame: "global" or "ego" — frame the returned velocity is expressed in.
        ref_pose: pose defining the output ego frame (defaults to pose1).

    Returns: (2,) or (N, 2) velocity in m/s.

    Timestamps are microseconds; dt in seconds is (t1 - t0) / 1e6. nuScenes sweep
    intervals are not exactly uniform, so a hardcoded 1/12 s puts a systematic
    error into every derived quantity — and TTC is the most safety-relevant
    number the monitor reports.
    """
    dt = (t1_us - t0_us) / 1e6
    if dt <= 0:
        raise ValueError(f"non-positive dt ({dt}s); frames must be time-ordered")

    p0 = np.atleast_2d(np.asarray(p0_ego, dtype=np.float64))
    p1 = np.atleast_2d(np.asarray(p1_ego, dtype=np.float64))
    g0 = ego_to_global(p0, pose0)
    g1 = ego_to_global(p1, pose1)
    v_global = (g1 - g0) / dt

    if in_frame == "global":
        out = v_global
    elif in_frame == "ego":
        R = Quaternion((ref_pose or pose1)["rotation"]).rotation_matrix[:2, :2]
        out = v_global @ R          # global->ego rotation is R^T, applied as v @ R
    else:
        raise ValueError(f"in_frame must be 'ego' or 'global', got {in_frame!r}")

    return out[0] if np.asarray(p0_ego).ndim == 1 else out


def time_to_collision(
    position_ego: np.ndarray,
    velocity_ego: np.ndarray,
) -> float:
    """
    Time to collision along the ego-object line of sight.

    Args:
        position_ego: (2,) object position in the ego frame, metres.
        velocity_ego: (2,) object velocity RELATIVE to the ego vehicle, m/s.
            Must already be ego-motion compensated — pass the output of
            velocity_from_track minus the ego's own velocity, or a relative
            velocity computed directly in the ego frame.

    Returns: seconds to collision, or inf if the object is not closing.

    ttc = range / closing_speed, where closing_speed is the component of relative
    velocity along the unit vector from ego to object, sign-flipped so that
    approaching is positive.

    The guard matters: for a receding object closing_speed is negative, and an
    unguarded divide returns a large NEGATIVE ttc which trivially passes a
    `ttc < threshold` test — firing a collision warning for every car driving
    away from you. Return inf instead.
    """
    p = np.asarray(position_ego, dtype=np.float64)
    v = np.asarray(velocity_ego, dtype=np.float64)
    rng = float(np.linalg.norm(p))
    if rng < 1e-6:
        return 0.0
    closing = -float(np.dot(v, p / rng))   # positive when approaching
    if closing <= 1e-6:
        return float("inf")
    return rng / closing
