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

WHAT RADAR IS BAD AT (be honest in the write-up):
    Sparse (tens to a few hundred returns/sweep vs ~30k LiDAR points), no height
    information worth using, heavy multipath clutter, and poor angular resolution.
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
        filter_invalid: drop returns flagged invalid / Doppler-ambiguous / likely
            false alarms. See the filtering guidance in `filter_radar_points`.

    Returns:
        (N, 6) float32 array of [x, y, z, rcs, vx_comp, vy_comp], all in the ego
        frame of THIS sample. N varies per frame (typically ~100-400 across all
        five sensors) — this is a ragged, per-frame quantity, so callers must not
        assume a fixed N.

    IMPLEMENTATION NOTES
    --------------------
    For each channel:
      1. `sd = nusc.get("sample_data", sample["data"][channel])`
      2. Load the point cloud. `RadarPointCloud.from_file(path)` gives an (18, N)
         array in `.points` — note it is CHANNELS-FIRST, so transpose it.
         The devkit applies its own default filtering at class level; to control
         it yourself, set `RadarPointCloud.disable_filters()` and filter here.
      3. Get `cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])`.
      4. Transform to ego:
             R = Quaternion(cs["rotation"]).rotation_matrix   # (3,3)
             t = np.array(cs["translation"])                  # (3,)
             xyz_ego = xyz_sensor @ R.T + t
      5. **Velocity is a direction, not a position** — rotate it but do NOT add
         the translation:
             v_ego = v_sensor @ R.T
         (vx_comp/vy_comp are 2D; lift to 3D with vz=0 before rotating, then keep
         the x,y components.) Getting this wrong is the classic radar-fusion bug:
         translating a velocity vector silently corrupts every motion feature, and
         the model will still train — just worse. Assert that the speed magnitude
         is preserved by the rotation as a cheap self-check.
      6. Concatenate all five sensors' points into one (N, 6) array.
    """
    raise NotImplementedError("P10-1")


def filter_radar_points(points_18: np.ndarray) -> np.ndarray:
    """
    Drop unreliable radar returns from a raw (N, 18) array.

    Args:
        points_18: raw radar returns, ego- or sensor-frame, 18 channels.

    Returns:
        (M, 18) subset with M <= N.

    RECOMMENDED FILTER (tune, then justify in the write-up):
        - invalid_state (col 14) == 0            — sensor marked the return valid
        - ambig_state   (col 11) == 3            — Doppler measurement unambiguous
        - pdh0          (col 15) <= 3            — low false-alarm probability

    Be careful here. Aggressive filtering is tempting because it produces clean
    visualisations, but each dropped return is a potential object the fusion
    model can no longer see. Radar is already sparse. Log the mean surviving
    point count per frame before and after — if you are discarding more than
    about half, the filter is too strict.
    """
    raise NotImplementedError("P10-1")


def rasterize_radar_bev(
    points: np.ndarray,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
) -> np.ndarray:
    """
    Rasterize ego-frame radar returns into a dense multi-channel BEV grid.

    Args:
        points: (N, 6) [x, y, z, rcs, vx, vy] in the ego frame (from load_radar_points).
        xbound: (min, max, cell_size) along ego x — must match the camera BEV grid.
        ybound: (min, max, cell_size) along ego y — must match the camera BEV grid.

    Returns:
        (5, X, Y) float32 grid with channels RADAR_BEV_CHANNELS:
            occupancy — number of returns in the cell (or log1p of it)
            rcs       — mean radar cross-section of returns in the cell
            vx, vy    — mean compensated velocity components
            speed     — mean sqrt(vx^2 + vy^2)
        Empty cells are 0 in every channel.

    IMPLEMENTATION NOTES
    --------------------
    Mirror the indexing convention in `LiftSplatShoot.splat` exactly, or the radar
    grid will be transposed or half-cell-shifted relative to the camera grid and
    fusion will quietly learn nothing:

        ix = ((x - xbound[0]) / xbound[2]).astype(int)
        iy = ((y - ybound[0]) / ybound[2]).astype(int)
        keep = (0 <= ix < X) & (0 <= iy < Y)

    Accumulate with `np.add.at(grid[c], (ix, iy), value)` (plain fancy indexing
    does NOT accumulate on duplicate indices — it keeps only the last write, which
    is exactly wrong when several returns share a cell). Then divide the sum
    channels by the occupancy count where count > 0 to get means.

    SPARSITY WARNING: with a 0.8 m cell over a 128x128 grid you have 16,384 cells
    and maybe 200 returns, so >98% of the grid is empty. A plain conv stack will
    mostly convolve zeros. Two mitigations worth trying, and worth reporting as an
    ablation: (a) dilate the occupancy channel with a small max-pool so each return
    covers a 3x3 neighbourhood, reflecting radar's true angular uncertainty;
    (b) add a signed-distance-to-nearest-return channel. Do the simple version
    first and measure it before adding either.
    """
    raise NotImplementedError("P10-2")


def radar_bev_for_sample(
    nusc,
    data_root: str | Path,
    sample: dict,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    channels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Convenience wrapper: load + filter + rasterize in one call.

    Returns: (5, X, Y) float32 radar BEV grid, ready to stack with the camera BEV.
    Used by NuScenesBEVDataset when cfg["use_radar"] is set.
    """
    raise NotImplementedError("P10-2")
