import json
from types import SimpleNamespace

import numpy as np

import mcp_server.perception_tools as perception_tools


class FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


class FakeStore:
    def __init__(self):
        self.loaded = SimpleNamespace(
            frame_id="frame-1",
            scene_name="scene-001",
            frame_idx=2,
            timestamp=123456789,
        )
        self.records = {"frame-1": self.loaded}

    def list_scenes(self):
        return [
            {
                "name": "scene-001",
                "description": "synthetic clear weather scene",
                "num_frames": 3,
            }
        ]

    def load_frame(self, scene_name, frame_idx):
        assert scene_name == "scene-001"
        assert frame_idx == 2
        return self.loaded

    def get(self, frame_id):
        return self.records[frame_id]


class FakeRegistry:
    def __init__(self, perception):
        self.store = FakeStore()
        self.perception = perception
        self.calls = []

    def run_perception(self, frame_id):
        self.calls.append(frame_id)
        return self.perception


def register_tools(monkeypatch, perception):
    mcp = FakeMCP()
    registry = FakeRegistry(perception)
    monkeypatch.setattr(perception_tools, "get_registry", lambda: registry)
    perception_tools.register_all_tools(mcp)
    return mcp.tools, registry


def parse(result):
    return json.loads(result)


def make_mask(*, lane=False, drivable=False, crossing=False, walkway=False):
    mask = np.zeros((9, 12), dtype=np.int64)
    if walkway:
        mask[0:2, :] = 4

    # Ahead region is rows 6:9 and columns 3:9, for 18 pixels.
    if drivable:
        mask[6:8, 3:6] = 1  # 6/18 = 33.3%, above the 30% threshold.
    if lane:
        mask[8, 3] = 2  # 1/18 = 5.6%, above the 2% threshold.
    if crossing:
        mask[8, 4] = 3  # 1/18 = 5.6%, above the 1% threshold.
    return mask


def make_perception(
    *,
    boxes=None,
    scores=None,
    labels=None,
    seg_mask=None,
    bev_boxes=None,
    bev_scores=None,
    bev_labels=None,
):
    boxes = [] if boxes is None else boxes
    scores = [] if scores is None else scores
    labels = [] if labels is None else labels
    bev_boxes = [] if bev_boxes is None else bev_boxes
    bev_scores = [] if bev_scores is None else bev_scores
    bev_labels = [] if bev_labels is None else bev_labels

    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "seg_mask": make_mask() if seg_mask is None else seg_mask,
        "bev_boxes": np.asarray(bev_boxes, dtype=np.float32).reshape(-1, 5),
        "bev_scores": np.asarray(bev_scores, dtype=np.float32),
        "bev_labels": np.asarray(bev_labels, dtype=np.int64),
        "bev_seg": np.zeros((4, 4), dtype=np.int64),
    }


def test_register_all_tools_exposes_expected_mcp_surface(monkeypatch):
    tools, _ = register_tools(monkeypatch, make_perception())

    assert set(tools) == {
        "list_scenes",
        "load_frame",
        "detect_objects",
        "segment_scene",
        "bev_map",
        "check_lane_switch_safety",
        "check_turn_clearance",
        "check_obstacle_stop",
        "check_pedestrian_crossing",
        "estimate_following_distance",
        "scene_summary",
    }


def test_list_scenes_and_load_frame_return_agent_friendly_json(monkeypatch):
    tools, _ = register_tools(monkeypatch, make_perception())

    assert parse(tools["list_scenes"]()) == [
        {
            "name": "scene-001",
            "description": "synthetic clear weather scene",
            "num_frames": 3,
        }
    ]
    assert parse(tools["load_frame"]("scene-001", 2)) == {
        "frame_id": "frame-1",
        "scene_name": "scene-001",
        "frame_idx": 2,
        "timestamp_us": 123456789,
    }


def test_detect_objects_counts_rounds_and_sorts_by_confidence(monkeypatch):
    perception = make_perception(
        boxes=[
            [0.04, 1.05, 10.06, 20.04],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        scores=[0.6129, 0.951, 0.333],
        labels=[0, 1, 0],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["detect_objects"]("frame-1"))

    assert result["counts"] == {"car": 2, "pedestrian": 1, "cyclist": 0}
    assert [box["class"] for box in result["boxes"]] == [
        "pedestrian",
        "car",
        "car",
    ]
    assert result["boxes"][1] == {
        "class": "car",
        "score": 0.613,
        "x1": 0.0,
        "y1": 1.0,
        "x2": 10.1,
        "y2": 20.0,
    }


def test_segment_scene_reports_coverage_and_ahead_flags(monkeypatch):
    mask = make_mask(lane=True, drivable=True, crossing=True, walkway=True)
    tools, _ = register_tools(monkeypatch, make_perception(seg_mask=mask))

    result = parse(tools["segment_scene"]("frame-1"))

    assert result["coverage_pct"]["walkway"] > 1.0
    assert result["drivable_ahead"] is True
    assert result["lane_markings_visible"] is True
    assert result["ped_crossing_ahead"] is True
    assert result["walkway_present"] is True


def test_bev_map_uses_ego_coordinates_and_sorts_by_range(monkeypatch):
    perception = make_perception(
        bev_boxes=[
            [12.0, 5.0, 4.4, 1.8, 0.0],
            [3.0, 4.0, 0.8, 0.5, 0.0],
        ],
        bev_scores=[0.8123, 0.9001],
        bev_labels=[0, 2],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["bev_map"]("frame-1"))

    assert result["grid_extent_m"] == {"x": [0, 51.2], "y": [-25.6, 25.6]}
    assert [obj["class"] for obj in result["objects"]] == ["cyclist", "car"]
    assert result["objects"][0]["range_m"] == 5.0
    assert result["objects"][0]["bearing_deg"] == 53.1


def test_lane_switch_safety_checks_correct_lateral_band_and_boundaries(monkeypatch):
    perception = make_perception(
        seg_mask=make_mask(lane=True),
        bev_boxes=[
            [0.0, -3.0, 4.0, 2.0, 0.0],   # ignored: not forward.
            [20.0, -1.9, 4.0, 2.0, 0.0],  # ignored: current lane.
            [40.0, -6.0, 4.0, 2.0, 0.0],  # included: left boundary.
            [12.0, 2.0, 4.0, 2.0, 0.0],   # included: right boundary.
            [41.0, 3.0, 4.0, 2.0, 0.0],   # ignored: beyond lookahead.
        ],
        bev_scores=[0.5, 0.6, 0.7, 0.8, 0.9],
        bev_labels=[0, 0, 2, 0, 0],
    )
    tools, _ = register_tools(monkeypatch, perception)

    left = parse(tools["check_lane_switch_safety"]("frame-1", "left"))
    right = parse(tools["check_lane_switch_safety"]("frame-1", "right"))

    assert left["safe"] is False
    assert left["lane_marking_present"] is True
    assert left["obstacles_in_target_lane"] == [
        {"class": "cyclist", "x_m": 40.0, "y_m": -6.0, "score": 0.7}
    ]
    assert right["safe"] is False
    assert right["obstacles_in_target_lane"] == [
        {"class": "car", "x_m": 12.0, "y_m": 2.0, "score": 0.8}
    ]


def test_lane_switch_can_be_safe_even_when_lane_marking_is_absent(monkeypatch):
    tools, _ = register_tools(monkeypatch, make_perception(seg_mask=make_mask()))

    result = parse(tools["check_lane_switch_safety"]("frame-1", "left"))

    assert result["safe"] is True
    assert result["lane_marking_present"] is False
    assert result["obstacles_in_target_lane"] == []


def test_turn_clearance_combines_bev_pedestrian_and_crossing_hazards(monkeypatch):
    perception = make_perception(
        labels=[1],
        seg_mask=make_mask(crossing=True),
        bev_boxes=[[15.0, -10.0, 4.0, 2.0, 0.0]],
        bev_scores=[0.91],
        bev_labels=[0],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["check_turn_clearance"]("frame-1", "left"))

    assert result["clear"] is False
    assert result["approaching_vehicles"] == [
        {"class": "car", "x_m": 15.0, "y_m": -10.0, "score": 0.91}
    ]
    assert result["pedestrians_detected"] == 1
    assert result["ped_crossing_active"] is True
    assert "NOT CLEAR" in result["recommendation"]


def test_turn_clearance_is_clear_when_no_hazards_are_present(monkeypatch):
    tools, _ = register_tools(monkeypatch, make_perception())

    result = parse(tools["check_turn_clearance"]("frame-1", "right"))

    assert result["clear"] is True
    assert result["direction"] == "right"
    assert result["approaching_vehicles"] == []
    assert result["pedestrians_detected"] == 0
    assert result["ped_crossing_active"] is False
    assert "Turn right CLEAR" in result["recommendation"]
    assert "no obstacles detected" in result["recommendation"]


def test_obstacle_stop_finds_nearest_current_lane_obstacle_only(monkeypatch):
    perception = make_perception(
        bev_boxes=[
            [18.0, 1.5, 4.0, 2.0, 0.0],
            [6.0, -2.0, 0.8, 0.5, 0.0],
            [5.0, 2.1, 4.0, 2.0, 0.0],
            [25.0, 0.0, 4.0, 2.0, 0.0],
        ],
        bev_scores=[0.7, 0.95, 0.99, 0.8],
        bev_labels=[0, 2, 0, 0],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["check_obstacle_stop"]("frame-1"))

    assert result["stop_required"] is True
    assert result["nearest_obstacle_class"] == "cyclist"
    assert result["nearest_obstacle_distance_m"] == 6.0
    assert [obj["x_m"] for obj in result["all_obstacles_ahead"]] == [6.0, 18.0]


def test_pedestrian_crossing_distinguishes_caution_from_wait(monkeypatch):
    crossing_only = make_perception(seg_mask=make_mask(crossing=True))
    tools, _ = register_tools(monkeypatch, crossing_only)

    caution = parse(tools["check_pedestrian_crossing"]("frame-1"))

    assert caution["crossing_ahead"] is True
    assert caution["pedestrians_in_scene"] == 0
    assert caution["safe_to_proceed"] is True
    assert "CAUTION" in caution["recommendation"]

    with_pedestrian = make_perception(labels=[1], seg_mask=make_mask(crossing=True))
    tools, _ = register_tools(monkeypatch, with_pedestrian)

    wait = parse(tools["check_pedestrian_crossing"]("frame-1"))

    assert wait["pedestrians_on_crossing_estimate"] is True
    assert wait["safe_to_proceed"] is False
    assert "WAIT" in wait["recommendation"]


def test_estimate_following_distance_uses_nearest_car_in_current_lane(monkeypatch):
    perception = make_perception(
        bev_boxes=[
            [4.0, 0.0, 4.0, 2.0, 0.0],
            [8.0, 1.5, 0.8, 0.5, 0.0],
            [12.0, -2.0, 4.0, 2.0, 0.0],
            [3.0, 3.0, 4.0, 2.0, 0.0],
        ],
        bev_scores=[0.66, 0.99, 0.88, 0.77],
        bev_labels=[0, 2, 0, 0],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["estimate_following_distance"]("frame-1"))

    assert result["vehicle_ahead"] is True
    assert result["distance_m"] == 4.0
    assert result["lateral_offset_m"] == 0.0
    assert result["score"] == 0.66
    assert "TOO CLOSE" in result["recommendation"]
    assert "4.0 m" in result["recommendation"]


def test_estimate_following_distance_reports_no_vehicle_when_only_non_cars_match(monkeypatch):
    perception = make_perception(
        bev_boxes=[[8.0, 0.0, 0.8, 0.5, 0.0]],
        bev_scores=[0.99],
        bev_labels=[2],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["estimate_following_distance"]("frame-1"))

    assert result == {
        "vehicle_ahead": False,
        "distance_m": None,
        "lateral_offset_m": None,
        "score": None,
        "recommendation": "No vehicle detected ahead in current lane.",
    }


def test_scene_summary_combines_counts_nearest_zones_and_safety_flags(monkeypatch):
    perception = make_perception(
        labels=[0, 1, 2, 0],
        seg_mask=make_mask(drivable=True, crossing=True),
        bev_boxes=[
            [10.0, 0.0, 4.0, 2.0, 0.0],
            [8.0, -3.0, 4.0, 2.0, 0.0],
            [12.0, 4.0, 0.8, 0.5, 0.0],
        ],
        bev_scores=[0.8, 0.7, 0.9],
        bev_labels=[0, 0, 2],
    )
    tools, _ = register_tools(monkeypatch, perception)

    result = parse(tools["scene_summary"]("frame-1"))

    assert result["frame"] == {
        "scene": "scene-001",
        "frame_idx": 2,
        "timestamp_us": 123456789,
    }
    assert result["detections_2d"] == {"car": 2, "pedestrian": 1, "cyclist": 1}
    assert result["detections_bev"] == {"car": 2, "pedestrian": 0, "cyclist": 1}
    assert result["nearest_objects"] == {
        "ahead": {"class": "car", "x_m": 10.0, "y_m": 0.0},
        "left_lane": {"class": "car", "x_m": 8.0, "y_m": -3.0},
        "right_lane": {"class": "cyclist", "x_m": 12.0, "y_m": 4.0},
    }
    assert result["safety_flags"] == {
        "left_lane_clear": False,
        "right_lane_clear": False,
        "drivable_ahead": True,
        "ped_crossing_ahead": True,
        "pedestrians_present": True,
    }
