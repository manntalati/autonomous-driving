"""Tests for P11 — calibration metrics, signals, introspection head, BEV matching."""
import numpy as np
import pytest
import torch

from evaluation.bev_matching import (
    box_ranges,
    centre_distance_matrix,
    compute_bev_map,
    match_by_centre_distance,
)
from evaluation.calibration import (
    _auroc,
    expected_calibration_error,
    failure_prediction_auroc,
    platt_scale,
    risk_coverage_curve,
    selective_summary,
    temperature_scale,
)
from models.uncertainty.introspection import IntrospectionHead, TrustScorer
from models.uncertainty.mc_dropout import active_dropout_p, enable_dropout
from models.uncertainty.signals import (
    FEATURE_NAMES,
    FeatureScaler,
    assemble_features,
    radar_camera_disagreement,
    temporal_instability,
)


class TestCalibration:
    def test_perfect_calibration_near_zero(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20000)
        y = (rng.uniform(0, 1, 20000) < p).astype(int)
        assert expected_calibration_error(p, y)[0] < 0.02

    def test_overconfidence_detected(self):
        rng = np.random.default_rng(1)
        p = np.full(5000, 0.9)
        y = (rng.uniform(0, 1, 5000) < 0.5).astype(int)
        assert expected_calibration_error(p, y)[0] == pytest.approx(0.4, abs=0.05)

    def test_empty_input(self):
        ece, _ = expected_calibration_error(np.zeros(0), np.zeros(0))
        assert np.isnan(ece)

    def test_auroc_extremes(self):
        assert _auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
        assert _auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0

    def test_auroc_ties_are_chance(self):
        assert _auroc(np.full(4, 0.5), np.array([0, 0, 1, 1])) == 0.5

    def test_auroc_single_class_is_nan(self):
        assert np.isnan(_auroc(np.array([0.1, 0.9]), np.array([1, 1])))

    def test_temperature_recovers_known_inflation(self):
        rng = np.random.default_rng(2)
        z = rng.normal(0, 2, 20000)
        y = (rng.uniform(0, 1, 20000) < 1 / (1 + np.exp(-z))).astype(int)
        assert temperature_scale(z * 3, y) == pytest.approx(3.0, rel=0.05)

    def test_platt_corrects_bias_shift(self):
        """A temperature alone cannot undo a prior shift; two parameters can."""
        rng = np.random.default_rng(3)
        z = rng.normal(0, 2, 20000)
        y = (rng.uniform(0, 1, 20000) < 1 / (1 + np.exp(-z))).astype(int)
        a, b = platt_scale(z + 2.0, y)          # injected bias of +2
        assert a > 0
        assert b == pytest.approx(-2.0, abs=0.3)

    def test_calibration_does_not_change_ranking(self):
        rng = np.random.default_rng(4)
        z = rng.normal(0, 2, 2000)
        y = (rng.uniform(0, 1, 2000) < 1 / (1 + np.exp(-z))).astype(int)
        a, b = platt_scale(z, y)
        before = _auroc(1 / (1 + np.exp(-z)), y)
        after = _auroc(1 / (1 + np.exp(-(a * z + b))), y)
        assert before == pytest.approx(after, abs=1e-9)

    def test_risk_decreases_with_lower_coverage(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(int)
        rc = risk_coverage_curve(p, y)
        assert rc["risk"][0] <= rc["risk"][-1]
        assert 0.0 <= rc["aurc"] <= 1.0

    def test_selective_summary_improves_error(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(int)
        s = selective_summary(p, y, 0.5)
        assert s["error_kept"] < s["error_all"]

    def test_failure_prediction_reports_baseline(self):
        m = failure_prediction_auroc(np.array([0.1, 0.9]), np.array([0, 1]))
        assert m["positive_rate"] == 0.5 and m["n"] == 2


class TestSignals:
    def test_feature_width_matches_names(self):
        f = assemble_features(np.zeros((3, 5)), np.zeros(3), np.zeros(3, dtype=int))
        assert f.shape == (3, len(FEATURE_NAMES))

    def test_missing_signals_flagged_not_zeroed(self):
        """'no radar nearby' and 'radar unavailable' must be distinguishable."""
        no_radar = assemble_features(np.zeros((1, 5)), np.zeros(1), np.zeros(1, dtype=int))
        with_radar = assemble_features(np.zeros((1, 5)), np.zeros(1), np.zeros(1, dtype=int),
                                       radar=np.zeros((1, 5)))
        i = FEATURE_NAMES.index("radar_avail")
        assert no_radar[0, i] == 0.0 and with_radar[0, i] == 1.0

    def test_empty_detections(self):
        assert assemble_features(np.zeros((0, 5)), np.zeros(0), np.zeros(0, dtype=int)).shape[0] == 0

    def test_radar_corroboration(self):
        det = np.array([[10.0, 0.0, 4, 2, 0]])
        near = np.array([[10.2, 0.1, 0, 15.0, 1.0, 0.0]])
        assert radar_camera_disagreement(det, near)[0, 4] == 1.0
        assert radar_camera_disagreement(det, np.zeros((0, 6)))[0, 4] == 0.0

    def test_radar_far_return_not_corroborating(self):
        det = np.array([[10.0, 0.0, 4, 2, 0]])
        far = np.array([[40.0, 30.0, 0, 15.0, 0.0, 0.0]])
        assert radar_camera_disagreement(det, far)[0, 0] == 0

    def test_temporal_instability_bounds(self):
        stable = {0: [{"score": 0.9, "xy": (i, 0)} for i in range(5)]}
        gappy = {0: [None, None, {"score": 0.4, "xy": (0, 0)}, None, None]}
        assert 0.0 <= temporal_instability(stable)[0] < 0.2
        assert temporal_instability(gappy)[0] > 0.5

    def test_scaler_standardises(self):
        x = np.random.default_rng(0).normal(5, 3, (500, 4))
        out = FeatureScaler().fit_transform(x)
        assert np.allclose(out.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(out.std(axis=0), 1, atol=1e-4)

    def test_scaler_handles_constant_column(self):
        x = np.ones((10, 3))
        assert np.isfinite(FeatureScaler().fit_transform(x)).all()

    def test_scaler_requires_fit(self):
        with pytest.raises(RuntimeError):
            FeatureScaler().transform(np.zeros((2, 3)))

    def test_scaler_roundtrip(self):
        x = np.random.default_rng(1).normal(0, 1, (50, 4))
        s = FeatureScaler().fit(x)
        s2 = FeatureScaler().load_state_dict(s.state_dict())
        assert np.allclose(s.transform(x), s2.transform(x))


class TestIntrospection:
    def test_forward_shape(self):
        h = IntrospectionHead(in_features=len(FEATURE_NAMES))
        assert h(torch.randn(7, len(FEATURE_NAMES))).shape == (7,)

    def test_rejects_bad_rank(self):
        with pytest.raises(ValueError):
            IntrospectionHead()(torch.randn(3))

    def test_predict_proba_in_range(self):
        h = IntrospectionHead()
        p = h.predict_proba(torch.randn(20, len(FEATURE_NAMES)))
        assert ((p >= 0) & (p <= 1)).all()

    def test_calibration_is_monotonic(self):
        h = IntrospectionHead()
        x = torch.randn(50, len(FEATURE_NAMES))
        before = h.predict_proba(x)
        h.calib_a, h.calib_b = 0.5, -1.0
        after = h.predict_proba(x)
        assert (torch.argsort(before) == torch.argsort(after)).all()

    def test_trust_scorer_empty_frame_is_in_odd(self):
        """An empty road is not an untrustworthy frame."""
        t = TrustScorer(IntrospectionHead()).score_frame(
            np.zeros((0, 5)), np.zeros(0), np.zeros(0), np.zeros((0, len(FEATURE_NAMES))))
        assert t["in_odd"] and t["trust"] == 1.0

    def test_trust_scorer_output_contract(self):
        det = np.array([[10.0, 0.0, 4, 2, 0]])
        f = assemble_features(det, np.array([0.8]), np.array([0]))
        out = TrustScorer(IntrospectionHead()).score_frame(det, [0.8], [0], f)
        assert set(out) == {"trust", "in_odd", "reason", "per_detection"}
        assert 0.0 <= out["trust"] <= 1.0
        assert out["per_detection"][0]["class"] == "car"

    def test_risk_weight_favours_close_vulnerable_in_path(self):
        s = TrustScorer(IntrospectionHead())
        close_ped = s._weights(np.array([[5.0, 0.0, 1, 1, 0]]), [1])
        far_car = s._weights(np.array([[45.0, 20.0, 4, 2, 0]]), [0])
        assert close_ped[0] > far_car[0]


class TestMCDropout:
    def test_enable_dropout_counts(self):
        m = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2, 2), torch.nn.Dropout2d(0.1))
        m.eval()
        assert enable_dropout(m) == 2

    def test_enable_dropout_leaves_batchnorm_in_eval(self):
        """Phase 2's bug: model.train() also updates BN running stats on eval data."""
        m = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Dropout(0.5))
        m.eval()
        enable_dropout(m)
        assert not m[0].training and m[1].training

    def test_active_dropout_p(self):
        m = torch.nn.Sequential(torch.nn.Dropout(0.25))
        assert active_dropout_p(m) == [0.25]

    def test_stochastic_dropout_restores_state(self):
        """Leaving dropout enabled made the NEXT 'deterministic' forward random,
        which randomised the detection set and broke the P11 A/B comparison."""
        from models.uncertainty.mc_dropout import stochastic_dropout
        m = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(4))
        m.eval()
        with stochastic_dropout(m) as n:
            assert n == 1 and m[0].training and not m[1].training
        assert not m[0].training and not m[1].training

    def test_stochastic_dropout_restores_on_exception(self):
        from models.uncertainty.mc_dropout import stochastic_dropout
        m = torch.nn.Sequential(torch.nn.Dropout(0.5))
        m.eval()
        with pytest.raises(RuntimeError):
            with stochastic_dropout(m):
                raise RuntimeError("boom")
        assert not m[0].training

    def test_predictor_leaves_model_deterministic(self):
        """End-to-end: two forwards after MC sampling must agree exactly."""
        from models.uncertainty.mc_dropout import MCDropoutPredictor

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = torch.nn.Dropout(0.5)
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, x, return_raw=False, **kw):
                h = self.fc(self.drop(x))
                return [h.unsqueeze(1)], [h.unsqueeze(1)[..., :1].repeat(1, 1, 4)], torch.zeros(1, 4)

        m = Tiny().eval()
        x = torch.randn(2, 4)
        MCDropoutPredictor(m, num_samples=4)(x)
        with torch.no_grad():
            a = m(x, return_raw=True)[0][0]
            b = m(x, return_raw=True)[0][0]
        assert torch.equal(a, b), "model left stochastic after MC sampling"


class TestBEVMatching:
    def test_distance_matrix_shape(self):
        assert centre_distance_matrix(np.zeros((3, 5)), np.zeros((4, 5))).shape == (3, 4)

    def test_empty_inputs(self):
        assert centre_distance_matrix(np.zeros((0, 5)), np.zeros((2, 5))).shape == (0, 2)

    def test_match_within_threshold(self):
        tp, mg = match_by_centre_distance(np.array([[0.0, 0, 1, 1, 0]]), np.array([0.9]),
                                          np.array([[1.0, 0, 1, 1, 0]]), threshold=2.0)
        assert tp[0] and mg[0] == 0

    def test_match_beyond_threshold_fails(self):
        tp, _ = match_by_centre_distance(np.array([[0.0, 0, 1, 1, 0]]), np.array([0.9]),
                                         np.array([[10.0, 0, 1, 1, 0]]), threshold=2.0)
        assert not tp[0]

    def test_each_gt_matched_once(self):
        """Duplicate detections must count as false positives."""
        tp, _ = match_by_centre_distance(
            np.array([[0.0, 0, 1, 1, 0], [0.3, 0, 1, 1, 0]]), np.array([0.9, 0.8]),
            np.array([[0.0, 0, 1, 1, 0]]), threshold=2.0)
        assert tp.sum() == 1

    def test_greedy_prefers_higher_score(self):
        tp, _ = match_by_centre_distance(
            np.array([[5.0, 0, 1, 1, 0], [0.1, 0, 1, 1, 0]]), np.array([0.2, 0.95]),
            np.array([[0.0, 0, 1, 1, 0]]), threshold=2.0)
        assert tp[1] and not tp[0]

    def test_perfect_map(self):
        preds = [{"boxes": [[0, 0, 4, 2, 0]], "scores": [0.9], "labels": [0]}]
        gts = [{"boxes": [[0, 0, 4, 2, 0]], "labels": [0]}]
        assert compute_bev_map(preds, gts, 1)[0] == pytest.approx(1.0, abs=1e-6)

    def test_no_gt_gives_zero(self):
        preds = [{"boxes": [[0, 0, 4, 2, 0]], "scores": [0.9], "labels": [0]}]
        gts = [{"boxes": [], "labels": []}]
        assert compute_bev_map(preds, gts, 1)[0] == 0.0

    def test_box_ranges(self):
        assert box_ranges(np.array([[3.0, 4.0, 1, 1, 0]]))[0] == pytest.approx(5.0)
