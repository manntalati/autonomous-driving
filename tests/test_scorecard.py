"""Tests for P14-2 — the unified scorecard (no data or checkpoints needed)."""
import csv
import json
import math
import zlib
from pathlib import Path

import pytest
from scipy.stats import t as student_t

from evaluation.scorecard import (CELL_ORDER, CI_SUFFIX, META_COLUMNS, RANGE_NAMES,
                                  bev_seed_cells, build_row, csv_record,
                                  detector_seed_cells, expand_ckpts, main,
                                  parse_seeds, summarize, upsert_csv,
                                  validate_seed_args)
from utils.provenance import config_hash, git_info

LOGS = Path(__file__).resolve().parent.parent / "logs"


def _audit(seen=0.6, day=0.3, night=0.1, miniday=0.3):
    return {name: {"mAP": v, "AP": [v, v, v], "num_frames": 10}
            for name, v in (("seen_day", seen), ("unseen_day", day),
                            ("unseen_night", night), ("unseen_miniday", miniday))}


def _foreign(phone=0.28, dash=0.23, action=0.02):
    return {cam: {"mAP": v, "AP": [v, v, v], "frames": 10}
            for cam, v in (("phone", phone), ("dashcam", dash), ("action_cam", action))}


def _bev(day=0.14, night=0.01):
    def cell(m):
        return {"mAP": m, "AP": [m, m, m], "num_frames": 5, "num_gt": 50,
                "buckets": {r: {"mAP": m / (i + 1), "AP": [m] * 3, "num_gt": 10}
                            for i, r in enumerate(RANGE_NAMES)}}
    return {"unseen_day": cell(day), "unseen_night": cell(night)}


def _row(det, bev, name="exp"):
    return build_row(name, det, bev, components={"detector": {"config": "c.yaml", "config_hash": "h",
                                                              "ckpts": ["a.pt"] * len(det)}},
                     data={}, settings={"foreign_fov_normalize": True}, details={},
                     wall_time_s=1.0, git={"sha": "0123456789abcdef", "dirty": False})


class TestSummarize:
    def test_empty(self):
        s = summarize([])
        assert s["n"] == 0 and s["mean"] is None and s["ci95"] is None

    def test_single_seed_has_no_interval(self):
        s = summarize([0.3])
        assert s["n"] == 1 and s["mean"] == pytest.approx(0.3)
        assert s["std"] is None and s["ci95"] is None

    def test_three_seeds_t_interval(self):
        vals = [0.28, 0.30, 0.29]
        s = summarize(vals)
        mean = sum(vals) / 3
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / 2)
        half = student_t.ppf(0.975, 2) * sd / math.sqrt(3)
        assert s["n"] == 3
        assert s["mean"] == pytest.approx(mean)
        assert s["std"] == pytest.approx(sd)
        assert s["ci95"] == pytest.approx([mean - half, mean + half])

    def test_missing_values_are_skipped_not_zeroed(self):
        s = summarize([0.2, None, 0.4])
        assert s["n"] == 2 and s["mean"] == pytest.approx(0.3)
        assert s["values"] == [0.2, None, 0.4]


class TestCells:
    def test_detector_cells(self):
        cells, details = detector_seed_cells(_audit(day=0.3, night=0.1), _foreign())
        assert cells["det.night_gap_rel"] == pytest.approx((0.3 - 0.1) / 0.3)
        assert cells["det.foreign.mean.mAP"] == pytest.approx((0.28 + 0.23 + 0.02) / 3)
        assert details["AP"]["det.unseen_night"] == [0.1, 0.1, 0.1]
        assert details["frames"]["det.foreign.phone"] == 10

    def test_gap_undefined_when_day_is_zero(self):
        cells, _ = detector_seed_cells(_audit(day=0.0), _foreign())
        assert "det.night_gap_rel" not in cells

    def test_foreign_mean_needs_every_camera(self):
        f = _foreign()
        del f["action_cam"]
        cells, _ = detector_seed_cells(_audit(), f)
        assert "det.foreign.mean.mAP" not in cells

    def test_bev_cells(self):
        cells, details = bev_seed_cells(_bev(day=0.14))
        assert cells["bev.unseen_day.mAP"] == pytest.approx(0.14)
        assert cells["bev.unseen_day.far.mAP"] == pytest.approx(0.14 / 3)
        assert details["num_gt"]["bev.unseen_night.far"] == 10

    def test_range_names_match_radar_ablation(self):
        from evaluation.radar_ablation import RANGE_BUCKETS
        assert tuple(name for name, _, _ in RANGE_BUCKETS) == RANGE_NAMES

    def test_gap_is_paired_per_seed(self):
        # Seed A: good day, bad night. Seed B: the reverse. Pairing within a seed
        # gives gaps of 0.5 and -1.0; pairing across seeds would give neither.
        a, _ = detector_seed_cells(_audit(day=0.4, night=0.2), _foreign())
        b, _ = detector_seed_cells(_audit(day=0.2, night=0.4), _foreign())
        row = _row([a, b], [])
        assert row["cells"]["det.night_gap_rel"]["values"] == pytest.approx([0.5, -1.0])


class TestRow:
    def test_every_cell_present_in_order(self):
        det = [detector_seed_cells(_audit(), _foreign())[0]]
        row = _row(det, [bev_seed_cells(_bev())[0]])
        assert tuple(row["cells"])[:len(CELL_ORDER)] == CELL_ORDER
        assert all(row["cells"][k]["n"] == 1 for k in CELL_ORDER)

    def test_skipped_bev_leaves_empty_cells(self):
        row = _row([detector_seed_cells(_audit(), _foreign())[0]], [])
        assert row["cells"]["bev.unseen_night.far.mAP"]["n"] == 0
        assert row["cells"]["bev.unseen_night.far.mAP"]["mean"] is None

    def test_row_is_json_serialisable(self):
        row = _row([detector_seed_cells(_audit(), _foreign())[0]], [bev_seed_cells(_bev())[0]])
        json.dumps(row)

    def test_committed_logs_reproduce_readme_numbers(self):
        """The evaluators' real output shapes flatten into the numbers the README quotes."""
        audit = json.load(open(LOGS / "day_night_audit.json"))["cells"]
        foreign = json.load(open(LOGS / "foreign_camera_eval_norm.json"))["cells"]
        radar = json.load(open(LOGS / "radar_ablation.json"))
        det, _ = detector_seed_cells(audit, foreign)
        bev, _ = bev_seed_cells({"unseen_day": radar["camera/unseen_day"],
                                 "unseen_night": radar["camera/unseen_night"]})
        cells = _row([det], [bev])["cells"]
        assert cells["det.unseen_day.mAP"]["mean"] == pytest.approx(0.2853, abs=1e-4)
        assert cells["det.unseen_night.mAP"]["mean"] == pytest.approx(0.0951, abs=1e-4)
        assert cells["det.night_gap_rel"]["mean"] == pytest.approx(0.667, abs=1e-3)
        assert cells["det.foreign.mean.mAP"]["mean"] == pytest.approx(0.1775, abs=1e-4)
        assert cells["bev.unseen_day.far.mAP"]["mean"] == pytest.approx(0.0124, abs=1e-4)
        assert all(cells[k]["n"] == 1 for k in CELL_ORDER)


class TestSeeds:
    def test_parse(self):
        assert parse_seeds("0,1,2") == [0, 1, 2]
        assert parse_seeds("") == []
        with pytest.raises(ValueError):
            parse_seeds("1,1")

    def test_expand_template(self):
        assert expand_ckpts("ck_s{seed}.pt", [0, 1]) == (["ck_s0.pt", "ck_s1.pt"], [0, 1])

    def test_plain_path_is_one_unknown_seed(self):
        assert expand_ckpts("ck.pt", [0, 1, 2]) == (["ck.pt"], [None])

    def test_template_without_seeds_rejected(self):
        with pytest.raises(ValueError):
            expand_ckpts("ck_s{seed}.pt", [])

    def test_seeds_without_any_template_rejected(self):
        with pytest.raises(ValueError):
            validate_seed_args([0, 1, 2], ["det.pt", "bev.pt"])
        validate_seed_args([0, 1, 2], ["det_s{seed}.pt", "bev.pt"])

    def test_cli_rejects_before_touching_data(self):
        with pytest.raises(SystemExit):
            main(["--name", "x", "--seeds", "0,1,2"])
        with pytest.raises(SystemExit):
            main(["--name", "has space"])


class TestCsv:
    def test_write_then_upsert_replaces(self, tmp_path):
        path = tmp_path / "scorecard.csv"
        det = [detector_seed_cells(_audit(day=0.3), _foreign())[0]]
        upsert_csv(path, _row(det, [], name="baseline"))
        upsert_csv(path, _row(det, [], name="r1"))
        det2 = [detector_seed_cells(_audit(day=0.5), _foreign())[0]]
        header = upsert_csv(path, _row(det2, [], name="baseline"))
        rows = list(csv.DictReader(path.open()))
        assert [r["experiment"] for r in rows] == ["r1", "baseline"]
        assert float(rows[1]["det.unseen_day.mAP"]) == pytest.approx(0.5)
        assert header[:len(META_COLUMNS)] == list(META_COLUMNS)
        assert header[len(META_COLUMNS):len(META_COLUMNS) + 2] == [
            CELL_ORDER[0], CELL_ORDER[0] + CI_SUFFIX]

    def test_new_cell_appends_column_without_reordering(self, tmp_path):
        path = tmp_path / "scorecard.csv"
        det = [detector_seed_cells(_audit(), _foreign())[0]]
        before = upsert_csv(path, _row(det, [], name="a"))
        extra = dict(det[0], **{"det.zz_new_cell": 0.5})
        after = upsert_csv(path, _row([extra], [], name="b"))
        assert after[:len(before)] == before
        assert after[-2:] == ["det.zz_new_cell", "det.zz_new_cell" + CI_SUFFIX]
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["det.zz_new_cell"] == ""

    def test_ci_half_width(self):
        det = [detector_seed_cells(_audit(day=d), _foreign())[0] for d in (0.28, 0.30, 0.29)]
        rec = csv_record(_row(det, []))
        ci = _row(det, [])["cells"]["det.unseen_day.mAP"]["ci95"]
        assert float(rec["det.unseen_day.mAP" + CI_SUFFIX]) == pytest.approx((ci[1] - ci[0]) / 2, abs=1e-6)
        assert rec["git_sha"] == "0123456789ab"


class TestProvenance:
    def test_config_hash_ignores_key_order(self):
        assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})

    def test_config_hash_sees_value_changes(self):
        assert config_hash({"lr": 0.001}) != config_hash({"lr": 0.002})
        assert len(config_hash({})) == 12

    def test_git_info_shape(self):
        info = git_info()
        assert set(info) == {"sha", "dirty"}
        assert isinstance(info["sha"], str)


class TestForeignFrameSeed:
    def test_seed_is_process_independent(self):
        # hash() of a str is salted per process; the seed must not be.
        from evaluation.foreign_camera_eval import frame_seed
        assert frame_seed("abc") == zlib.crc32(b"abc") == 891568578
        assert 0 <= frame_seed("e3d495d4ac534d54b321f50006683844") < 2 ** 32


class TestRunWiring:
    """run_scorecard + main end to end, with the evaluators stubbed out."""

    def test_three_seed_run_writes_json_and_csv(self, tmp_path, monkeypatch):
        import evaluation.day_night_audit as dna
        import evaluation.foreign_camera_eval as fce
        import evaluation.radar_ablation as ra
        import models.detection.train_detector as td
        import nuscenes.nuscenes

        for s in (0, 1, 2):
            (tmp_path / f"det_s{s}.pt").touch()
        (tmp_path / "bev.pt").touch()

        cells = [(n, None, tmp_path, {f"scene-{n}"}) for n in
                 ("seen_day", "unseen_day", "unseen_night", "unseen_miniday")]
        day = {"det_s0.pt": 0.28, "det_s1.pt": 0.30, "det_s2.pt": 0.29}
        seen_foreign = []

        monkeypatch.setattr(nuscenes.nuscenes, "NuScenes", lambda **kw: object())
        monkeypatch.setattr(dna, "audit_cells", lambda *a: (cells, {"scene-x": {"night": True}}))
        monkeypatch.setattr(td, "load_detector", lambda cfg, ckpt, dev: Path(ckpt).name)
        monkeypatch.setattr(dna, "evaluate_audit_cells",
                            lambda model, cfg, dev, c: _audit(day=day[model], night=0.1))

        def fake_foreign(model, cfg, dev, nusc, root, scenes, cams, fov_normalize, include_native):
            seen_foreign.append((scenes, fov_normalize, include_native))
            return _foreign()
        monkeypatch.setattr(fce, "evaluate_foreign", fake_foreign)
        monkeypatch.setattr(ra, "load_model", lambda cfg, ckpt, dev: "bev")
        monkeypatch.setattr(ra, "build_loader", lambda *a: type("L", (), {"dataset": [0, 0]})())
        monkeypatch.setattr(ra, "evaluate_cell", lambda *a, **k: _bev()["unseen_day"])

        out_dir, csv_path = tmp_path / "out", tmp_path / "scorecard.csv"
        main(["--name", "wiring", "--det-ckpt", str(tmp_path / "det_s{seed}.pt"),
              "--bev-ckpt", str(tmp_path / "bev.pt"), "--seeds", "0,1,2",
              "--out-dir", str(out_dir), "--csv", str(csv_path)])

        row = json.load(open(out_dir / "wiring.json"))
        assert row["components"]["detector"]["seeds"] == [0, 1, 2]
        assert row["components"]["bev"]["seeds"] == [None]
        assert row["cells"]["det.unseen_day.mAP"]["n"] == 3
        assert row["cells"]["det.unseen_day.mAP"]["mean"] == pytest.approx(0.29)
        assert row["cells"]["bev.unseen_night.far.mAP"]["n"] == 1
        assert row["settings"]["foreign_fov_normalize"] is True
        # foreign cameras reuse the unseen_day frames and skip the native control
        assert seen_foreign == [({"scene-unseen_day"}, True, False)] * 3
        assert row["data"]["scenes"]["unseen_night"] == ["scene-unseen_night"]
        rec = next(csv.DictReader(csv_path.open()))
        assert rec["experiment"] == "wiring" and rec["det_n"] == "3" and rec["bev_n"] == "1"
