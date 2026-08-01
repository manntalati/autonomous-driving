"""
P10-1 / P10-2 — Radar point-cloud loading and BEV rasterization.

WHY RADAR (the Phase 10 thesis):
    Phase 9 measured a 67% mAP collapse from day to night. A camera infers depth
    and motion from *appearance*, so when photons run out, both collapse together.
    Radar measures range and radial velocity *directly*, by timing a returned
    radio pulse and reading its Doppler shift. Neither depends on illumination:
    a radar return at midnight is identical to one at noon.

    That makes radar the natural instrument for closing an illumination-induced
    domain gap — and the falsifiable Phase 10 claim: fusing radar should improve
    night mAP substantially more than it improves day mAP. If radar helps both
    equally, it is just adding capacity, not addressing the ODD problem.

WHAT RADAR IS BAD AT (stated honestly in the write-up):
    Sparse (tens to a few hundred returns/sweep vs ~30k LiDAR points), no usable
    height information, heavy multipath clutter, and poor angular resolution.
    Radar will not localise a pedestrian's silhouette; it will tell you something
    solid is at 23.4 m closing at 6 m/s. Fusion, not replacement.

nuScenes radar format (18 channels per point, see nuscenes.utils.data_classes):
    0,1,2   x, y, z          — position in the RADAR SENSOR frame (metres)
    3       dyn_prop         — dynamic property (0=moving, 1=stationary, ...)
    4       id               — tracking id assigned by the sensor
    5       rcs              — radar cross-section (dBsm); reflectivity proxy
    6,7     vx, vy           — raw radial velocity
    8,9     vx_comp, vy_comp — EGO-MOTION-COMPENSATED velocity (use these)
    10      is_quality_valid
    11      ambig_state      — 3 == unambiguous; others are Doppler-ambiguous
    12,13   x_rms, y_rms     — position uncertainty
    14      invalid_state    — 0 == valid
    15      pdh0             — false-alarm probability (0 == highest confidence)
    16,17   vx_rms, vy_rms
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

# The five radars ring the vehicle; together they give ~360 deg coverage.
RADAR_CHANNELS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

# Column indices into the raw 18-channel radar array.
IDX_XYZ = (0, 1, 2)
IDX_RCS = 5
IDX_VCOMP = (8, 9)
IDX_AMBIG = 11
IDX_INVALID = 14
IDX_PDH0 = 15

# Rasterized BEV channel layout produced by rasterize_radar_bev.
RADAR_BEV_CHANNELS = ["occupancy", "rcs", "vx", "vy", "speed"]
NUM_RADAR_BEV_CHANNELS = len(RADAR_BEV_CHANNELS)

# Per-channel divisors used to bring the rasterized channels onto a comparable
# scale before they hit the first conv. Fixed constants rather than BatchNorm:
# the radar grid is >98% zeros, so batch statistics are dominated by empty cells
# and drift badly between day and night frames.
RADAR_NORM = np.array([5.0, 20.0, 10.0, 10.0, 10.0], dtype=np.float32)


# invalid_state codes the sensor documents as VALID clusters, not just code 0.
# Measured over 20 mini frames (10,458 returns), keeping only code 0 retains 54%
# of returns, while these states together retain 82%. The important one is code 4,
# "valid cluster with low RCS" (13.1% of all returns): low RCS is exactly the
# radar signature of a pedestrian. Restricting to code 0 would systematically
# discard the returns the Phase 9 vulnerable-road-user argument depends on.
VALID_INVALID_STATES = (0, 4, 6, 7, 9, 12, 13, 14)


def filter_radar_points(points_18: np.ndarray, strict: bool = False) -> np.ndarray:
    """
    Drop unreliable radar returns from a raw (N, 18) array.

    Args:
        points_18: raw radar returns, 18 channels.
        strict: use the devkit default (invalid_state == 0 only). Provided so the
            filter itself can be ablated; the default here is deliberately wider.
    Returns: (M, 18) subset with M <= N.

    Filter:
        - invalid_state in VALID_INVALID_STATES  (or == 0 when strict)
        - ambig_state == 3     — Doppler measurement unambiguous (80.5% of returns)
        - pdh0 <= 3            — low false-alarm probability (94.3% of returns)

    Retention measured on nuScenes mini: 0.64 with the default, 0.47 with strict.
    Each dropped return is a potential object the fusion model can no longer see,
    and radar is already sparse — use `radar_point_stats` before tightening.
    """
    pts = np.asarray(points_18, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 18:
        raise ValueError(f"expected (N, 18) radar array, got {pts.shape}")
    if len(pts) == 0:
        return pts
    valid = (pts[:, IDX_INVALID] == 0) if strict else np.isin(pts[:, IDX_INVALID], VALID_INVALID_STATES)
    keep = valid & (pts[:, IDX_AMBIG] == 3) & (pts[:, IDX_PDH0] <= 3)
    return pts[keep]


def load_radar_points(
    nusc,
    data_root: str | Path,
    sample: dict,
    channels: Optional[List[str]] = None,
    filter_invalid: bool = True,
) -> np.ndarray:
    """
    Load all radar sweeps for one keyframe and return them in the EGO frame.

    Args:
        nusc: NuScenes instance.
        data_root: dataset root (the .pcd paths in sample_data are relative to it).
        sample: a nuScenes `sample` record (keyframe).
        channels: radar sensor names; defaults to all five in RADAR_CHANNELS.
        filter_invalid: apply `filter_radar_points`.

    Returns:
        (N, 6) float32 array of [x, y, z, rcs, vx_comp, vy_comp] in the ego frame
        of THIS sample. N varies per frame (typically ~100-400 across all five
        sensors), so callers must not assume a fixed N.
    """
    from nuscenes.utils.data_classes import RadarPointCloud

    channels = channels or RADAR_CHANNELS
    data_root = Path(data_root)
    collected: List[np.ndarray] = []

    for channel in channels:
        if channel not in sample["data"]:
            continue
        sd = nusc.get("sample_data", sample["data"][channel])
        path = data_root / sd["filename"]
        if not path.exists():
            continue

        # The devkit applies aggressive class-level filtering by default; disable
        # it so `filter_radar_points` is the single place filtering happens.
        RadarPointCloud.disable_filters()
        try:
            pc = RadarPointCloud.from_file(str(path))
        finally:
            RadarPointCloud.default_filters()
        raw = np.asarray(pc.points, dtype=np.float64).T          # (18, N) -> (N, 18)
        if len(raw) == 0:
            continue
        if filter_invalid:
            raw = filter_radar_points(raw)
        if len(raw) == 0:
            continue

        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        R = Quaternion(cs["rotation"]).rotation_matrix                 # (3, 3)
        t = np.asarray(cs["translation"], dtype=np.float64)            # (3,)

        xyz = raw[:, list(IDX_XYZ)] @ R.T + t                          # positions: rotate + translate

        # Velocity is a direction, not a position: rotate it, never translate it.
        # Translating a velocity vector corrupts every motion feature while still
        # training fine — it just trains worse, silently.
        v2 = raw[:, list(IDX_VCOMP)]
        v3 = np.concatenate([v2, np.zeros((len(v2), 1))], axis=1)
        v_ego = (v3 @ R.T)[:, :2]

        # Cheap self-check: a rotation is norm-preserving.
        if len(v3):
            assert np.allclose(np.linalg.norm(v3, axis=1),
                               np.linalg.norm(v3 @ R.T, axis=1), atol=1e-6), \
                "velocity rotation changed speed magnitude"

        rcs = raw[:, IDX_RCS : IDX_RCS + 1]
        collected.append(np.concatenate([xyz, rcs, v_ego], axis=1))

    if not collected:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(collected, axis=0).astype(np.float32)


def rasterize_radar_bev(
    points: np.ndarray,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    dilate: int = 1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Rasterize ego-frame radar returns into a dense multi-channel BEV grid.

    Args:
        points: (N, 6) [x, y, z, rcs, vx, vy] in the ego frame.
        xbound: (min, max, cell_size) along ego x — must match the camera BEV grid.
        ybound: (min, max, cell_size) along ego y — must match the camera BEV grid.
        dilate: half-width of a square dilation applied to every return, in cells.
            1 means each return paints a 3x3 neighbourhood, which reflects radar's
            true angular uncertainty and stops a conv stack from mostly convolving
            zeros. 0 disables it.
        normalize: divide channels by RADAR_NORM so they reach the first conv on
            comparable scales (occupancy ~counts, rcs in dBsm, velocity in m/s).

    Returns:
        (5, X, Y) float32 grid with channels RADAR_BEV_CHANNELS. Empty cells are 0.

    Indexing mirrors LiftSplatShoot.splat exactly; a half-cell offset or a
    transpose here would misalign the radar and camera grids and fusion would
    quietly learn nothing.
    """
    nx = int(round((xbound[1] - xbound[0]) / xbound[2]))
    ny = int(round((ybound[1] - ybound[0]) / ybound[2]))
    grid = np.zeros((NUM_RADAR_BEV_CHANNELS, nx, ny), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 6)
    if len(pts) == 0:
        return grid

    ix = np.floor((pts[:, 0] - xbound[0]) / xbound[2]).astype(np.int64)
    iy = np.floor((pts[:, 1] - ybound[0]) / ybound[2]).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, pts = ix[inside], iy[inside], pts[inside]
    if len(pts) == 0:
        return grid

    rcs = pts[:, 3]
    vx, vy = pts[:, 4], pts[:, 5]
    speed = np.sqrt(vx ** 2 + vy ** 2)

    offsets = range(-dilate, dilate + 1) if dilate > 0 else (0,)
    count = np.zeros((nx, ny), dtype=np.float64)
    sums = np.zeros((4, nx, ny), dtype=np.float64)   # rcs, vx, vy, speed

    for dx in offsets:
        for dy in offsets:
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            if not ok.any():
                continue
            # np.add.at accumulates on duplicate indices; plain fancy indexing
            # would keep only the last write, which is exactly wrong when several
            # returns share a cell.
            np.add.at(count, (jx[ok], jy[ok]), 1.0)
            np.add.at(sums[0], (jx[ok], jy[ok]), rcs[ok])
            np.add.at(sums[1], (jx[ok], jy[ok]), vx[ok])
            np.add.at(sums[2], (jx[ok], jy[ok]), vy[ok])
            np.add.at(sums[3], (jx[ok], jy[ok]), speed[ok])

    occupied = count > 0
    grid[0] = count.astype(np.float32)
    for c in range(4):
        chan = np.zeros((nx, ny), dtype=np.float64)
        chan[occupied] = sums[c][occupied] / count[occupied]     # mean, not sum
        grid[c + 1] = chan.astype(np.float32)

    if normalize:
        grid /= RADAR_NORM[:, None, None]
    return grid


def radar_bev_for_sample(
    nusc,
    data_root: str | Path,
    sample: dict,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    channels: Optional[List[str]] = None,
    dilate: int = 1,
) -> np.ndarray:
    """
    Convenience wrapper: load + filter + rasterize in one call.
    Returns: (5, X, Y) float32 radar BEV grid, ready to stack with the camera BEV.
    """
    pts = load_radar_points(nusc, data_root, sample, channels=channels)
    return rasterize_radar_bev(pts, xbound, ybound, dilate=dilate)


def radar_point_stats(
    nusc,
    data_root: str | Path,
    samples: List[dict],
    channels: Optional[List[str]] = None,
) -> dict:
    """
    Mean surviving radar returns per frame, before and after filtering.

    Use this before tightening `filter_radar_points`. If filtering discards more
    than about half the returns, it is too strict for a sensor this sparse.
    """
    raw_counts, kept_counts = [], []
    for sample in samples:
        raw = load_radar_points(nusc, data_root, sample, channels, filter_invalid=False)
        kept = load_radar_points(nusc, data_root, sample, channels, filter_invalid=True)
        raw_counts.append(len(raw))
        kept_counts.append(len(kept))
    return {
        "frames": len(samples),
        "mean_raw": float(np.mean(raw_counts)) if raw_counts else 0.0,
        "mean_kept": float(np.mean(kept_counts)) if kept_counts else 0.0,
        "retention": float(np.sum(kept_counts) / max(np.sum(raw_counts), 1)),
    }
