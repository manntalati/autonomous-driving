"""
Perception tool implementations for the MCP server (Phase 1).

Call register_all_tools(mcp) once from server.py.  Each tool is then
available in the agent's tool list via the FastMCP instance.

Tool inventory
──────────────
Core / foundation
  list_scenes()
  load_frame(scene_name, frame_idx)
  detect_objects(frame_id)
  segment_scene(frame_id)
  bev_map(frame_id)

High-level driving decisions (wrappers over the core tools' output)
  check_lane_switch_safety(frame_id, direction)
  check_turn_clearance(frame_id, direction)
  check_obstacle_stop(frame_id)
  check_pedestrian_crossing(frame_id)
  estimate_following_distance(frame_id)
  scene_summary(frame_id)

BEV coordinate convention (ego frame, matches the trained LSS model)
  x = forward from vehicle (0 → 51.2 m for single-camera front-only config)
  y = lateral (negative = left, positive = right)
"""
from __future__ import annotations

import json
import math
from typing import Literal

import numpy as np

from mcp_server.model_registry import get_registry
from mcp_server.scene_store import DETECT_CLASS_NAMES, SEG_CLASS_NAMES

# ── spatial thresholds (meters) ─────────────────────────────────────────────
_LANE_HALF_WIDTH    = 2.0    # current lane  |y| ≤ 2 m
_LANE_SWITCH_Y_MIN  = 2.0    # adjacent lane starts 2 m laterally from centre
_LANE_SWITCH_Y_MAX  = 6.0    # adjacent lane ends  6 m laterally
_LOOKAHEAD_M        = 40.0   # general BEV scan range
_STOP_LOOKAHEAD_M   = 20.0   # obstacle-stop trigger range
_FOLLOW_LOOKAHEAD_M = 50.0   # following-distance scan range
_TURN_LATERAL_M     = 15.0   # lateral extent for turn-clearance check

# ── segmentation "ahead" region ──────────────────────────────────────────────
# Bottom third of image, centre half of width  (nearest/centre field-of-view)
_SEG_ROW_FRAC = 2 / 3    # rows ≥ row_frac * H  are "near field"
_SEG_COL_LO   = 0.25     # centre column band start
_SEG_COL_HI   = 0.75     # centre column band end


# ── helpers ──────────────────────────────────────────────────────────────────

def _json(obj) -> str:
    return json.dumps(obj, indent=2)


def _seg_ahead_frac(seg_mask: np.ndarray, class_id: int) -> float:
    """Fraction of 'ahead' pixels belonging to class_id."""
    H, W = seg_mask.shape
    r0   = int(H * _SEG_ROW_FRAC)
    c0   = int(W * _SEG_COL_LO)
    c1   = int(W * _SEG_COL_HI)
    region = seg_mask[r0:, c0:c1]
    return float(np.mean(region == class_id))


def _bev_objects_in_band(
    bev_boxes:  np.ndarray,
    bev_labels: np.ndarray,
    bev_scores: np.ndarray,
    y_lo: float,
    y_hi: float,
    x_max: float,
) -> list[dict]:
    """Return BEV detections whose centre falls in the lateral band and forward range."""
    out = []
    for i, box in enumerate(bev_boxes):
        x, y = float(box[0]), float(box[1])
        if y_lo <= y <= y_hi and 0 < x <= x_max:
            out.append({
                "class":      DETECT_CLASS_NAMES[int(bev_labels[i])],
                "x_m":        round(x, 1),
                "y_m":        round(y, 1),
                "score":      round(float(bev_scores[i]), 3),
            })
    out.sort(key=lambda o: o["x_m"])
    return out


# ── registration entry point ─────────────────────────────────────────────────

def register_all_tools(mcp) -> None:
    """Bind all perception tools to the FastMCP instance `mcp`."""

    # =========================================================================
    # CORE TOOLS
    # =========================================================================

    @mcp.tool()
    def list_scenes() -> str:
        """
        List all nuScenes scenes available in the dataset.

        Returns:
            JSON array of {name, description, num_frames} objects, one per scene.
        """
        reg = get_registry()
        return _json(reg.store.list_scenes())

    @mcp.tool()
    def load_frame(scene_name: str, frame_idx: int) -> str:
        """
        Load a camera frame from a scene and register it server-side.
        Must be called before any other perception tool that takes frame_id.

        Args:
            scene_name: nuScenes scene name, e.g. "scene-0103".
            frame_idx:  0-based keyframe index within the scene.

        Returns:
            JSON with {frame_id, scene_name, frame_idx, timestamp_us}.
            Pass frame_id to all subsequent tool calls.
        """
        reg = get_registry()
        rec = reg.store.load_frame(scene_name, frame_idx)
        return _json({
            "frame_id":     rec.frame_id,
            "scene_name":   rec.scene_name,
            "frame_idx":    rec.frame_idx,
            "timestamp_us": rec.timestamp,
        })

    @mcp.tool()
    def detect_objects(frame_id: str) -> str:
        """
        Run 2D object detection on a loaded frame.

        Returns bounding boxes in pixel space (x1,y1,x2,y2) with class label
        and confidence score.  Classes: car (0), pedestrian (1), cyclist (2).

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {counts: {car, pedestrian, cyclist},
                       boxes: [{class, score, x1, y1, x2, y2}]}.
        """
        reg = get_registry()
        p   = reg.run_perception(frame_id)
        boxes, scores, labels = p["boxes"], p["scores"], p["labels"]

        counts     = {"car": 0, "pedestrian": 0, "cyclist": 0}
        detections = []
        for i in range(len(labels)):
            cls = DETECT_CLASS_NAMES[int(labels[i])]
            counts[cls] += 1
            b = boxes[i]
            detections.append({
                "class": cls,
                "score": round(float(scores[i]), 3),
                "x1":    round(float(b[0]), 1),
                "y1":    round(float(b[1]), 1),
                "x2":    round(float(b[2]), 1),
                "y2":    round(float(b[3]), 1),
            })
        detections.sort(key=lambda d: -d["score"])
        return _json({"counts": counts, "boxes": detections})

    @mcp.tool()
    def segment_scene(frame_id: str) -> str:
        """
        Run semantic segmentation on a loaded frame.

        Classifies every pixel into one of five classes:
        background, drivable, lane, ped_crossing, walkway.

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {coverage_pct: {class: pct}, drivable_ahead,
                       lane_markings_visible, ped_crossing_ahead, walkway_present}.
        """
        reg  = get_registry()
        p    = reg.run_perception(frame_id)
        mask = p["seg_mask"]

        coverage = {
            name: round(float(np.mean(mask == i)) * 100, 1)
            for i, name in enumerate(SEG_CLASS_NAMES)
        }
        return _json({
            "coverage_pct":          coverage,
            "drivable_ahead":        _seg_ahead_frac(mask, 1) > 0.30,
            "lane_markings_visible": _seg_ahead_frac(mask, 2) > 0.02,
            "ped_crossing_ahead":    _seg_ahead_frac(mask, 3) > 0.01,
            "walkway_present":       float(np.mean(mask == 4)) > 0.01,
        })

    @mcp.tool()
    def bev_map(frame_id: str) -> str:
        """
        Return a bird's-eye-view (top-down) spatial map of all detected objects.

        Coordinate convention — ego frame:
          x = forward (metres from vehicle front)
          y = lateral (negative = left, positive = right)

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {grid_extent_m, objects: [{class, score, x_m, y_m,
                       range_m, bearing_deg, length_m, width_m}]}.
        """
        reg = get_registry()
        p   = reg.run_perception(frame_id)
        boxes, scores, labels = p["bev_boxes"], p["bev_scores"], p["bev_labels"]

        objects = []
        for i in range(len(labels)):
            b   = boxes[i]   # [x, y, length, width, yaw]
            x, y = float(b[0]), float(b[1])
            objects.append({
                "class":       DETECT_CLASS_NAMES[int(labels[i])],
                "score":       round(float(scores[i]), 3),
                "x_m":         round(x, 1),
                "y_m":         round(y, 1),
                "range_m":     round(math.hypot(x, y), 1),
                "bearing_deg": round(math.degrees(math.atan2(y, x)), 1),
                "length_m":    round(float(b[2]), 1),
                "width_m":     round(float(b[3]), 1),
            })
        objects.sort(key=lambda o: o["range_m"])
        return _json({
            "grid_extent_m": {"x": [0, 51.2], "y": [-25.6, 25.6]},
            "objects":        objects,
        })

    # =========================================================================
    # HIGH-LEVEL DRIVING DECISION TOOLS
    # =========================================================================

    @mcp.tool()
    def check_lane_switch_safety(
        frame_id: str,
        direction: Literal["left", "right"],
    ) -> str:
        """
        Assess whether switching to the adjacent lane is safe.

        Uses BEV detections to check for vehicles and cyclists in the target
        lane band (2–6 m laterally) within 40 m, and segmentation to verify
        that lane markings are visible.

        Args:
            frame_id:  ID returned by load_frame.
            direction: "left" or "right".

        Returns:
            JSON with {safe, direction, obstacles_in_target_lane,
                       lane_marking_present, recommendation}.
        """
        reg = get_registry()
        p   = reg.run_perception(frame_id)

        if direction == "left":
            y_lo, y_hi = -_LANE_SWITCH_Y_MAX, -_LANE_SWITCH_Y_MIN
        else:
            y_lo, y_hi =  _LANE_SWITCH_Y_MIN,  _LANE_SWITCH_Y_MAX

        obstacles = _bev_objects_in_band(
            p["bev_boxes"], p["bev_labels"], p["bev_scores"],
            y_lo, y_hi, _LOOKAHEAD_M,
        )
        lane_marking = _seg_ahead_frac(p["seg_mask"], 2) > 0.02
        safe         = len(obstacles) == 0

        return _json({
            "safe":                    safe,
            "direction":               direction,
            "obstacles_in_target_lane": obstacles,
            "lane_marking_present":    lane_marking,
            "recommendation": (
                f"Lane switch {direction} SAFE — target lane is clear."
                if safe else
                f"Lane switch {direction} NOT SAFE — {len(obstacles)} obstacle(s) in target lane."
            ),
        })

    @mcp.tool()
    def check_turn_clearance(
        frame_id: str,
        direction: Literal["left", "right"],
    ) -> str:
        """
        Assess clearance for a turn at an intersection.

        Checks three hazards:
        - Vehicles/cyclists in the turn-side lateral band (BEV).
        - Pedestrians anywhere in the 2D detection output.
        - Active pedestrian crossing ahead (segmentation).

        Args:
            frame_id:  ID returned by load_frame.
            direction: "left" or "right".

        Returns:
            JSON with {clear, direction, approaching_vehicles,
                       pedestrians_detected, ped_crossing_active, recommendation}.
        """
        reg = get_registry()
        p   = reg.run_perception(frame_id)

        if direction == "left":
            y_lo, y_hi = -_TURN_LATERAL_M, -_LANE_SWITCH_Y_MIN
        else:
            y_lo, y_hi =  _LANE_SWITCH_Y_MIN, _TURN_LATERAL_M

        approaching  = _bev_objects_in_band(
            p["bev_boxes"], p["bev_labels"], p["bev_scores"],
            y_lo, y_hi, _LOOKAHEAD_M,
        )
        ped_count    = int(np.sum(p["labels"] == 1))
        ped_crossing = _seg_ahead_frac(p["seg_mask"], 3) > 0.01

        clear  = len(approaching) == 0 and ped_count == 0 and not ped_crossing
        issues = []
        if approaching:
            issues.append(f"{len(approaching)} vehicle(s) in turn path")
        if ped_count > 0:
            issues.append(f"{ped_count} pedestrian(s) detected")
        if ped_crossing:
            issues.append("active pedestrian crossing ahead")

        return _json({
            "clear":                clear,
            "direction":            direction,
            "approaching_vehicles": approaching,
            "pedestrians_detected": ped_count,
            "ped_crossing_active":  ped_crossing,
            "recommendation": (
                f"Turn {direction} CLEAR — no obstacles detected."
                if clear else
                f"Turn {direction} NOT CLEAR — {'; '.join(issues)}."
            ),
        })

    @mcp.tool()
    def check_obstacle_stop(frame_id: str) -> str:
        """
        Check whether the vehicle should stop due to an obstacle directly ahead.

        Scans the current lane (|y| ≤ 2 m) in BEV space for any object within
        20 m.  Note: this is obstacle-only — traffic lights and stop signs are
        not in the detection model's class set.

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {stop_required, nearest_obstacle_class,
                       nearest_obstacle_distance_m, all_obstacles_ahead,
                       recommendation}.
        """
        reg = get_registry()
        p   = reg.run_perception(frame_id)

        ahead = _bev_objects_in_band(
            p["bev_boxes"], p["bev_labels"], p["bev_scores"],
            -_LANE_HALF_WIDTH, _LANE_HALF_WIDTH, _STOP_LOOKAHEAD_M,
        )

        stop    = len(ahead) > 0
        nearest = ahead[0] if ahead else None

        return _json({
            "stop_required":               stop,
            "nearest_obstacle_class":      nearest["class"]  if nearest else None,
            "nearest_obstacle_distance_m": nearest["x_m"]    if nearest else None,
            "all_obstacles_ahead":         ahead,
            "recommendation": (
                f"STOP — {nearest['class']} {nearest['x_m']} m ahead."
                if stop else
                "Path clear — no obstacles within stop range."
            ),
        })

    @mcp.tool()
    def check_pedestrian_crossing(frame_id: str) -> str:
        """
        Detect pedestrian crossings ahead and whether pedestrians are using them.

        Uses segmentation (ped_crossing class) to detect a marked crossing in
        the vehicle's path, and 2D detection to count pedestrians in the scene.

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {crossing_ahead, crossing_coverage_pct,
                       pedestrians_in_scene, pedestrians_on_crossing_estimate,
                       safe_to_proceed, recommendation}.
        """
        reg  = get_registry()
        p    = reg.run_perception(frame_id)
        mask = p["seg_mask"]

        crossing_ahead    = _seg_ahead_frac(mask, 3) > 0.01
        crossing_coverage = round(_seg_ahead_frac(mask, 3) * 100, 1)
        ped_count         = int(np.sum(p["labels"] == 1))
        peds_on_crossing  = crossing_ahead and ped_count > 0
        safe              = not peds_on_crossing

        if not safe:
            rec_text = "WAIT — pedestrians detected on crossing ahead."
        elif crossing_ahead:
            rec_text = "CAUTION — crossing ahead but no pedestrians observed."
        else:
            rec_text = "No crossing detected — proceed normally."

        return _json({
            "crossing_ahead":                 crossing_ahead,
            "crossing_coverage_pct":          crossing_coverage,
            "pedestrians_in_scene":           ped_count,
            "pedestrians_on_crossing_estimate": peds_on_crossing,
            "safe_to_proceed":                safe,
            "recommendation":                 rec_text,
        })

    @mcp.tool()
    def estimate_following_distance(frame_id: str) -> str:
        """
        Estimate the distance to the nearest car directly ahead in the current lane.

        Uses BEV detections; only considers vehicles classified as 'car' with
        |y| ≤ 2 m (current lane) and x ≤ 50 m.

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {vehicle_ahead, distance_m, lateral_offset_m,
                       score, recommendation}.
        """
        reg = get_registry()
        p   = reg.run_perception(frame_id)
        boxes, scores, labels = p["bev_boxes"], p["bev_scores"], p["bev_labels"]

        candidates = []
        for i in range(len(labels)):
            if int(labels[i]) != 0:   # cars only
                continue
            x, y = float(boxes[i][0]), float(boxes[i][1])
            if abs(y) <= _LANE_HALF_WIDTH and 0 < x <= _FOLLOW_LOOKAHEAD_M:
                candidates.append({
                    "distance_m":   round(x, 1),
                    "lateral_m":    round(y, 1),
                    "score":        round(float(scores[i]), 3),
                })
        candidates.sort(key=lambda v: v["distance_m"])
        nearest = candidates[0] if candidates else None

        if nearest:
            d = nearest["distance_m"]
            if d < 5:
                rec_text = f"TOO CLOSE — {d} m. Brake immediately."
            elif d < 15:
                rec_text = f"CLOSE — {d} m. Reduce speed."
            else:
                rec_text = f"Following distance {d} m — adequate."
        else:
            rec_text = "No vehicle detected ahead in current lane."

        return _json({
            "vehicle_ahead":    nearest is not None,
            "distance_m":       nearest["distance_m"]  if nearest else None,
            "lateral_offset_m": nearest["lateral_m"]   if nearest else None,
            "score":            nearest["score"]        if nearest else None,
            "recommendation":   rec_text,
        })

    @mcp.tool()
    def scene_summary(frame_id: str) -> str:
        """
        Comprehensive scene summary combining all perception outputs.

        Runs the full pipeline (detection + segmentation + BEV) and synthesises
        counts, road surface coverage, nearest objects per zone, and key safety
        flags into a single JSON document the agent can reason over holistically.

        Args:
            frame_id: ID returned by load_frame.

        Returns:
            JSON with {frame, detections_2d, detections_bev, road_surface_pct,
                       nearest_objects, safety_flags}.
        """
        reg = get_registry()
        rec = reg.store.get(frame_id)
        p   = reg.run_perception(frame_id)

        # ── object counts ──────────────────────────────────────────────────
        counts_2d  = {"car": 0, "pedestrian": 0, "cyclist": 0}
        for lbl in p["labels"]:
            counts_2d[DETECT_CLASS_NAMES[int(lbl)]] += 1

        counts_bev = {"car": 0, "pedestrian": 0, "cyclist": 0}
        for lbl in p["bev_labels"]:
            counts_bev[DETECT_CLASS_NAMES[int(lbl)]] += 1

        # ── segmentation coverage ──────────────────────────────────────────
        mask     = p["seg_mask"]
        coverage = {
            name: round(float(np.mean(mask == i)) * 100, 1)
            for i, name in enumerate(SEG_CLASS_NAMES)
        }

        # ── nearest object per BEV zone ────────────────────────────────────
        def _nearest(y_lo, y_hi, x_max=_LOOKAHEAD_M):
            best = None
            for i, box in enumerate(p["bev_boxes"]):
                x, y = float(box[0]), float(box[1])
                if y_lo <= y <= y_hi and 0 < x <= x_max:
                    if best is None or x < best["x_m"]:
                        best = {
                            "class": DETECT_CLASS_NAMES[int(p["bev_labels"][i])],
                            "x_m":   round(x, 1),
                            "y_m":   round(y, 1),
                        }
            return best

        return _json({
            "frame": {
                "scene":        rec.scene_name,
                "frame_idx":    rec.frame_idx,
                "timestamp_us": rec.timestamp,
            },
            "detections_2d":  counts_2d,
            "detections_bev": counts_bev,
            "road_surface_pct": coverage,
            "nearest_objects": {
                "ahead":       _nearest(-_LANE_HALF_WIDTH,   _LANE_HALF_WIDTH),
                "left_lane":   _nearest(-_LANE_SWITCH_Y_MAX, -_LANE_SWITCH_Y_MIN),
                "right_lane":  _nearest( _LANE_SWITCH_Y_MIN,  _LANE_SWITCH_Y_MAX),
            },
            "safety_flags": {
                "left_lane_clear":     len(_bev_objects_in_band(
                    p["bev_boxes"], p["bev_labels"], p["bev_scores"],
                    -_LANE_SWITCH_Y_MAX, -_LANE_SWITCH_Y_MIN, _LOOKAHEAD_M,
                )) == 0,
                "right_lane_clear":    len(_bev_objects_in_band(
                    p["bev_boxes"], p["bev_labels"], p["bev_scores"],
                    _LANE_SWITCH_Y_MIN, _LANE_SWITCH_Y_MAX, _LOOKAHEAD_M,
                )) == 0,
                "drivable_ahead":      _seg_ahead_frac(mask, 1) > 0.30,
                "ped_crossing_ahead":  _seg_ahead_frac(mask, 3) > 0.01,
                "pedestrians_present": int(np.sum(p["labels"] == 1)) > 0,
            },
        })
