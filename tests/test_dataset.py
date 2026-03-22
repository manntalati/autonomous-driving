"""
Unit tests for data/dataset.py — mockable components.

Tests that require the full nuScenes data directory are skipped in CI via
the `nuscenes_data` mark.  The bulk of tests cover:

  - LABEL_MAP / CLASS_NAMES constants
  - collate_fn — variable-length box batching
  - NuScenesDetectionDataset._get_2d_boxes logic (via a white-box unit of the
    clipping + filtering rules, exercised directly without the nuScenes API)
  - NuScenesDetectionDataset with a fully mocked NuScenes API

No nuScenes data directory or GPU required for tests that do not carry the
`nuscenes_data` mark.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip guard — nuscenes devkit (pyquaternion) may not be installed in the
# system Python used for CI.  Any test class that imports from data.dataset
# must carry this mark.
# ---------------------------------------------------------------------------
try:
    import pyquaternion  # noqa: F401
    from nuscenes.nuscenes import NuScenes  # noqa: F401
    _NUSCENES_AVAILABLE = True
except ImportError:
    _NUSCENES_AVAILABLE = False

requires_nuscenes = pytest.mark.skipif(
    not _NUSCENES_AVAILABLE,
    reason="nuscenes-devkit (pyquaternion) not installed in this environment"
)


# ---------------------------------------------------------------------------
# LABEL_MAP / CLASS_NAMES
# ---------------------------------------------------------------------------

@requires_nuscenes
class TestLabelMap:

    def test_car_categories_map_to_0(self):
        from data.dataset import LABEL_MAP
        car_categories = [
            "vehicle.car",
            "vehicle.bus.bendy",
            "vehicle.bus.rigid",
            "vehicle.truck",
            "vehicle.trailer",
        ]
        for cat in car_categories:
            assert LABEL_MAP[cat] == 0, f"{cat} maps to {LABEL_MAP[cat]}, expected 0"

    def test_pedestrian_categories_map_to_1(self):
        from data.dataset import LABEL_MAP
        ped_categories = [
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.police_officer",
        ]
        for cat in ped_categories:
            assert LABEL_MAP[cat] == 1, f"{cat} maps to {LABEL_MAP[cat]}, expected 1"

    def test_cyclist_categories_map_to_2(self):
        from data.dataset import LABEL_MAP
        cyclist_categories = [
            "vehicle.motorcycle",
            "vehicle.bicycle",
        ]
        for cat in cyclist_categories:
            assert LABEL_MAP[cat] == 2, f"{cat} maps to {LABEL_MAP[cat]}, expected 2"

    def test_label_map_has_11_entries(self):
        """LABEL_MAP covers exactly 11 nuScenes categories (5 car + 4 ped + 2 cyclist)."""
        from data.dataset import LABEL_MAP
        assert len(LABEL_MAP) == 11, f"Expected 11 entries, got {len(LABEL_MAP)}"

    def test_label_values_are_0_1_or_2(self):
        from data.dataset import LABEL_MAP
        for cat, label in LABEL_MAP.items():
            assert label in {0, 1, 2}, \
                f"Category '{cat}' has label {label}, expected 0/1/2"

    def test_class_names_has_three_entries(self):
        from data.dataset import CLASS_NAMES
        assert len(CLASS_NAMES) == 3

    def test_class_names_ordered_by_label(self):
        from data.dataset import CLASS_NAMES
        assert CLASS_NAMES[0] == "car"
        assert CLASS_NAMES[1] == "pedestrian"
        assert CLASS_NAMES[2] == "cyclist"

    def test_unknown_category_not_in_label_map(self):
        """Categories outside the 3-class taxonomy must not appear in LABEL_MAP."""
        from data.dataset import LABEL_MAP
        excluded = [
            "vehicle.emergency.ambulance",
            "movable_object.barrier",
            "static_object.bicycle_rack",
            "animal",
        ]
        for cat in excluded:
            assert cat not in LABEL_MAP, f"'{cat}' should not be in LABEL_MAP"


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

@requires_nuscenes
class TestCollateFn:
    """Tests for the custom collate_fn that handles variable-length box lists."""

    def _make_item(self, num_boxes: int, h: int = 4, w: int = 4, num_classes: int = 3):
        """Build a (image_tensor, targets_dict) pair with a given number of boxes."""
        torch.manual_seed(num_boxes)
        image = torch.randn(3, h, w)
        if num_boxes > 0:
            boxes = torch.rand(num_boxes, 4)
            # Ensure x2 > x1, y2 > y1
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2].abs() + 0.01
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3].abs() + 0.01
            labels = torch.randint(0, num_classes, (num_boxes,))
        else:
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.long)
        targets = {
            'boxes': boxes,
            'labels': labels,
            'meta': {'sample_token': f'tok_{num_boxes}', 'camera': 'CAM_FRONT'},
        }
        return image, targets

    def test_images_are_stacked_into_single_tensor(self):
        from data.dataset import collate_fn
        batch = [self._make_item(2), self._make_item(3), self._make_item(1)]
        images, targets = collate_fn(batch)
        assert isinstance(images, torch.Tensor)
        assert images.shape == (3, 3, 4, 4)

    def test_targets_is_a_list(self):
        """Targets must remain a list (not stacked) to support variable box counts."""
        from data.dataset import collate_fn
        batch = [self._make_item(2), self._make_item(4)]
        images, targets = collate_fn(batch)
        assert isinstance(targets, list)

    def test_targets_list_length_matches_batch_size(self):
        from data.dataset import collate_fn
        batch = [self._make_item(i) for i in range(5)]
        images, targets = collate_fn(batch)
        assert len(targets) == 5

    def test_each_target_has_boxes_key(self):
        from data.dataset import collate_fn
        batch = [self._make_item(2), self._make_item(0)]
        _, targets = collate_fn(batch)
        for t in targets:
            assert 'boxes' in t

    def test_each_target_has_labels_key(self):
        from data.dataset import collate_fn
        batch = [self._make_item(3)]
        _, targets = collate_fn(batch)
        assert 'labels' in targets[0]

    def test_variable_box_counts_per_image(self):
        """Batch with different box counts must not crash (no stacking of boxes)."""
        from data.dataset import collate_fn
        batch = [self._make_item(0), self._make_item(5), self._make_item(2)]
        images, targets = collate_fn(batch)
        assert targets[0]['boxes'].shape == (0, 4)
        assert targets[1]['boxes'].shape == (5, 4)
        assert targets[2]['boxes'].shape == (2, 4)

    def test_empty_box_item_does_not_crash(self):
        """An image with zero annotations must produce a (0, 4) box tensor."""
        from data.dataset import collate_fn
        batch = [self._make_item(0), self._make_item(0)]
        images, targets = collate_fn(batch)
        assert images.shape[0] == 2
        for t in targets:
            assert t['boxes'].shape == (0, 4)

    def test_single_item_batch(self):
        from data.dataset import collate_fn
        batch = [self._make_item(3)]
        images, targets = collate_fn(batch)
        assert images.shape == (1, 3, 4, 4)
        assert len(targets) == 1

    def test_image_tensor_dtype_preserved(self):
        """collate_fn must not change image dtype."""
        from data.dataset import collate_fn
        batch = [self._make_item(1)]
        images, _ = collate_fn(batch)
        assert images.dtype == torch.float32

    def test_boxes_dtype_is_float32(self):
        from data.dataset import collate_fn
        batch = [self._make_item(3)]
        _, targets = collate_fn(batch)
        assert targets[0]['boxes'].dtype == torch.float32

    def test_labels_dtype_is_long(self):
        from data.dataset import collate_fn
        batch = [self._make_item(3)]
        _, targets = collate_fn(batch)
        assert targets[0]['labels'].dtype == torch.long

    def test_meta_dict_preserved(self):
        """Meta dict (sample_token, camera) must pass through unchanged."""
        from data.dataset import collate_fn
        batch = [self._make_item(1)]
        _, targets = collate_fn(batch)
        assert 'meta' in targets[0]
        assert 'sample_token' in targets[0]['meta']
        assert 'camera' in targets[0]['meta']

    def test_large_batch_does_not_crash(self):
        """Batch of 16 items must not crash or OOM on CPU."""
        from data.dataset import collate_fn
        batch = [self._make_item(i % 6) for i in range(16)]
        images, targets = collate_fn(batch)
        assert images.shape[0] == 16
        assert len(targets) == 16


# ---------------------------------------------------------------------------
# _get_2d_boxes logic — white-box unit tests via mocked NuScenes
# ---------------------------------------------------------------------------

@requires_nuscenes
class TestGet2dBoxesLogic:
    """
    Tests for NuScenesDetectionDataset._get_2d_boxes, exercised with a mock
    NuScenes API so no data directory is needed.

    The function:
      1. Calls nusc.get('sample_data', sd_token) to get calibrated_sensor_token
      2. Calls nusc.get('calibrated_sensor', token) to get camera_intrinsic K
      3. Calls nusc.get_sample_data(sd_token, box_vis_level=...) to get box list
      4. For each box: project corners with view_points(K, normalize=True)
      5. Clip, filter by area (min 2x2), filter by LABEL_MAP

    We mock the NuScenes API and use a known K to verify the clipping logic.
    """

    def _make_mock_box(self, name: str, corners_2d: np.ndarray):
        """Build a mock Box object with .name and .corners() returning a 3xN array.

        corners_2d: (2, N) array of [x; y] coordinates after projection.
        We embed these as the first two rows of a 3xN array (z=1 for normalize=True).
        """
        box = MagicMock()
        box.name = name
        # corners() returns (3, 8); after view_points with normalize=True → (3, 8)
        # where first 2 rows are pixel coords
        corners_3d = np.ones((3, corners_2d.shape[1]))
        corners_3d[0] = corners_2d[0]
        corners_3d[1] = corners_2d[1]
        box.corners.return_value = corners_3d
        return box

    def _build_dataset_with_mocks(self, boxes_raw):
        """Build a NuScenesDetectionDataset backed by a fully mocked NuScenes.

        Returns the dataset instance (not yet calling any sample loading).
        """
        from data.dataset import NuScenesDetectionDataset

        mock_nusc = MagicMock()

        # Mock _build_index to return an empty list so __init__ succeeds without real data
        with patch.object(NuScenesDetectionDataset, '_build_index', return_value=[]):
            ds = NuScenesDetectionDataset(
                nusc=mock_nusc,
                data_root='/fake/path',
                split='val',
            )

        # Set up the mocks for _get_2d_boxes
        mock_nusc.get.side_effect = lambda table, token: (
            {'calibrated_sensor_token': 'cs_tok'} if table == 'sample_data' else
            {'camera_intrinsic': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}  # identity K
        )
        mock_nusc.get_sample_data.return_value = (None, boxes_raw, None)
        return ds, mock_nusc

    def test_known_category_is_returned(self):
        """A box with a category in LABEL_MAP must appear in the output."""
        corners_2d = np.array([[10, 20, 30, 40, 10, 20, 30, 40],
                                [10, 10, 50, 50, 50, 50, 10, 10]], dtype=float)
        mock_box = self._make_mock_box("vehicle.car", corners_2d)

        ds, _ = self._build_dataset_with_mocks([mock_box])

        with patch('data.dataset.view_points', return_value=np.vstack([corners_2d, np.ones((1, 8))])):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)

        assert len(boxes_2d) == 1
        assert labels[0] == 0  # vehicle.car → class 0

    def test_unknown_category_is_filtered_out(self):
        """A box whose category is not in LABEL_MAP must be silently skipped."""
        corners_2d = np.array([[10, 20, 30, 40, 10, 20, 30, 40],
                                [10, 10, 50, 50, 50, 50, 10, 10]], dtype=float)
        mock_box = self._make_mock_box("movable_object.barrier", corners_2d)

        ds, _ = self._build_dataset_with_mocks([mock_box])

        with patch('data.dataset.view_points', return_value=np.vstack([corners_2d, np.ones((1, 8))])):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)

        assert len(boxes_2d) == 0
        assert len(labels) == 0

    def test_degenerate_box_filtered_by_min_size(self):
        """A box whose projected 2D extents are < 2 pixels must be dropped."""
        # Project all corners to nearly the same point: width < 2, height < 2
        corners_2d = np.array([[100, 100.5, 100, 100.5, 100, 100.5, 100, 100.5],
                                [200, 200, 200.5, 200.5, 200, 200, 200.5, 200.5]], dtype=float)
        mock_box = self._make_mock_box("vehicle.car", corners_2d)

        ds, _ = self._build_dataset_with_mocks([mock_box])

        with patch('data.dataset.view_points', return_value=np.vstack([corners_2d, np.ones((1, 8))])):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)

        assert len(boxes_2d) == 0, "Degenerate box should have been filtered by size check"

    def test_coordinates_clipped_to_image_bounds(self):
        """Projected corners outside image bounds must be clipped before returning."""
        # Corners extend beyond the image in x
        corners_2d = np.array([[-50, 1700, 1700, -50, -50, 1700, 1700, -50],
                                [100, 100,  200,  200, 100,  100,  200,  200]], dtype=float)
        mock_box = self._make_mock_box("vehicle.car", corners_2d)
        img_w, img_h = 1600, 900

        ds, _ = self._build_dataset_with_mocks([mock_box])

        with patch('data.dataset.view_points', return_value=np.vstack([corners_2d, np.ones((1, 8))])):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=img_w, img_h=img_h)

        assert len(boxes_2d) == 1
        x1, y1, x2, y2 = boxes_2d[0]
        assert x1 >= 0 and x2 <= img_w, f"x coords not clipped: [{x1}, {x2}]"
        assert y1 >= 0 and y2 <= img_h, f"y coords not clipped: [{y1}, {y2}]"

    def test_empty_box_list_returns_empty(self):
        """When there are no GT boxes, both output lists must be empty."""
        ds, _ = self._build_dataset_with_mocks([])
        boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)
        assert boxes_2d == []
        assert labels == []

    def test_multiple_classes_in_one_sample(self):
        """Multiple boxes of different classes must all be returned in order."""
        def _make_large_box(name, x_offset):
            cx = x_offset
            corners_2d = np.array([[cx, cx + 100, cx + 100, cx, cx, cx + 100, cx + 100, cx],
                                    [100, 100, 200, 200, 100, 100, 200, 200]], dtype=float)
            return self._make_mock_box(name, corners_2d), corners_2d

        box_car, c2d_car = _make_large_box("vehicle.car", 50)
        box_ped, c2d_ped = _make_large_box("human.pedestrian.adult", 300)
        box_cyc, c2d_cyc = _make_large_box("vehicle.bicycle", 600)

        ds, _ = self._build_dataset_with_mocks([box_car, box_ped, box_cyc])

        projected = {
            id(box_car): np.vstack([c2d_car, np.ones((1, 8))]),
            id(box_ped): np.vstack([c2d_ped, np.ones((1, 8))]),
            id(box_cyc): np.vstack([c2d_cyc, np.ones((1, 8))]),
        }

        call_idx = [0]
        boxes_in_order = [box_car, box_ped, box_cyc]
        projections_in_order = [
            np.vstack([c2d_car, np.ones((1, 8))]),
            np.vstack([c2d_ped, np.ones((1, 8))]),
            np.vstack([c2d_cyc, np.ones((1, 8))]),
        ]

        def fake_view_points(corners, K, normalize):
            idx = call_idx[0]
            call_idx[0] += 1
            return projections_in_order[idx]

        with patch('data.dataset.view_points', side_effect=fake_view_points):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)

        assert len(boxes_2d) == 3
        assert labels == [0, 1, 2]  # car, pedestrian, cyclist

    def test_pedestrian_category_label_is_1(self):
        """Pedestrian box must return label=1."""
        corners_2d = np.array([[100, 200, 200, 100, 100, 200, 200, 100],
                                [100, 100, 200, 200, 100, 100, 200, 200]], dtype=float)
        mock_box = self._make_mock_box("human.pedestrian.adult", corners_2d)

        ds, _ = self._build_dataset_with_mocks([mock_box])

        with patch('data.dataset.view_points', return_value=np.vstack([corners_2d, np.ones((1, 8))])):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)

        assert len(labels) == 1
        assert labels[0] == 1

    def test_output_boxes_have_four_coordinates(self):
        """Each returned box must have exactly 4 coordinates [x1, y1, x2, y2]."""
        corners_2d = np.array([[50, 150, 150, 50, 50, 150, 150, 50],
                                [50, 50, 150, 150, 50, 50, 150, 150]], dtype=float)
        mock_box = self._make_mock_box("vehicle.car", corners_2d)

        ds, _ = self._build_dataset_with_mocks([mock_box])

        with patch('data.dataset.view_points', return_value=np.vstack([corners_2d, np.ones((1, 8))])):
            boxes_2d, labels = ds._get_2d_boxes('sd_tok', img_w=1600, img_h=900)

        assert len(boxes_2d) == 1
        assert len(boxes_2d[0]) == 4


# ---------------------------------------------------------------------------
# NuScenesDetectionDataset — mocked full __getitem__
# ---------------------------------------------------------------------------

@requires_nuscenes
class TestNuScenesDatasetMocked:
    """
    Full __getitem__ test with a mocked NuScenes API and a synthetic PIL image.
    """

    @pytest.fixture
    def mocked_dataset(self, tmp_path):
        """
        Build a NuScenesDetectionDataset with a fully mocked NuScenes API.
        The mock returns one sample with a single car box.
        """
        from data.dataset import NuScenesDetectionDataset
        from PIL import Image

        # Create a fake image file
        img_dir = tmp_path / "samples" / "CAM_FRONT"
        img_dir.mkdir(parents=True)
        fake_img_path = img_dir / "fake.jpg"
        Image.new("RGB", (1600, 900), color=(128, 128, 128)).save(fake_img_path)

        mock_nusc = MagicMock()

        # sample_data record
        mock_nusc.get.side_effect = lambda table, _token: {
            'sample_data': {
                'filename': 'samples/CAM_FRONT/fake.jpg',
                'width': 1600,
                'height': 900,
                'calibrated_sensor_token': 'cs_tok',
            },
            'calibrated_sensor': {
                'camera_intrinsic': [
                    [1266.0, 0.0, 800.0],
                    [0.0, 1266.0, 450.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            'sample': {
                'data': {'CAM_FRONT': 'sd_tok'},
                'next': '',
            },
        }[table]

        # get_sample_data returns an empty box list for simplicity
        mock_nusc.get_sample_data.return_value = (None, [], None)

        # Patch _build_index to return a single (sample_token, camera) pair
        with patch.object(NuScenesDetectionDataset, '_build_index',
                          return_value=[('sample_tok', 'CAM_FRONT')]):
            ds = NuScenesDetectionDataset(
                nusc=mock_nusc,
                data_root=str(tmp_path),
                split='val',
            )
        return ds

    def test_len_matches_index_length(self, mocked_dataset):
        assert len(mocked_dataset) == 1

    def test_getitem_returns_tuple_of_two(self, mocked_dataset):
        item = mocked_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_image_tensor_shape(self, mocked_dataset):
        image, _ = mocked_dataset[0]
        assert image.shape == (3, 448, 800), \
            f"Expected (3, 448, 800), got {image.shape}"

    def test_image_tensor_dtype(self, mocked_dataset):
        image, _ = mocked_dataset[0]
        assert image.dtype == torch.float32

    def test_targets_has_required_keys(self, mocked_dataset):
        _, targets = mocked_dataset[0]
        assert 'boxes' in targets
        assert 'labels' in targets
        assert 'meta' in targets

    def test_empty_annotations_produce_zero_box_tensor(self, mocked_dataset):
        """When there are no GT boxes, boxes tensor must be (0, 4)."""
        _, targets = mocked_dataset[0]
        assert targets['boxes'].shape == (0, 4)
        assert targets['labels'].shape == (0,)

    def test_boxes_dtype_is_float32(self, mocked_dataset):
        _, targets = mocked_dataset[0]
        assert targets['boxes'].dtype == torch.float32

    def test_labels_dtype_is_long(self, mocked_dataset):
        _, targets = mocked_dataset[0]
        assert targets['labels'].dtype == torch.long

    def test_meta_contains_sample_token(self, mocked_dataset):
        _, targets = mocked_dataset[0]
        assert targets['meta']['sample_token'] == 'sample_tok'

    def test_meta_contains_camera_name(self, mocked_dataset):
        _, targets = mocked_dataset[0]
        assert targets['meta']['camera'] == 'CAM_FRONT'
