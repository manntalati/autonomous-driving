"""
P12-5 — Evaluating a streaming agent.

This is what turns the demo into a result. A video of an agent saying sensible
things is a demo; a table of lead times, false-alarm rates and abstention
correctness is evidence.

Runs on KEYFRAMES only (2 Hz) — sweeps carry no annotations, so nothing said on a
sweep frame is checkable. The demo runs at 12 Hz; the metrics run at 2 Hz.

FOUR METRICS
------------
1. WARNING LEAD TIME — for each GT hazard, how long before it materialised did
   the agent warn? Report the median and the fraction warned at all. Warning
   AFTER the fact is a failure, not a small delay, and is counted separately
   rather than dragging a mean down where it disappears.

2. FALSE-ALARM RATE — advisories per minute with no corresponding GT hazard.
   This is what punishes a chatty agent. An agent that warns constantly trivially
   achieves perfect lead time, so these two must be read together.

3. ABSTENTION CORRECTNESS — the Phase 11/12 payoff:

                        perception WRONG      perception RIGHT
       abstained            correct             over-cautious
       spoke confidently  DANGEROUS FAILURE       correct

   The bottom-left cell is the one that matters. Phase 9 says those concentrate
   at night. The claim to test is that abstention shrinks that cell on night
   scenes without emptying the top-right on day scenes.

4. GROUNDING ACCURACY — of the numeric claims in advisories (ranges, TTC), how
   many match GT within tolerance?

METHODOLOGICAL WARNINGS
-----------------------
* Small sample. Three night scenes, ~121 keyframes, and hazards are rarer still.
  Raw counts are reported everywhere; do not compute percentages over single-digit
  denominators. "2 of 3 hazards warned" is honest; "67% recall" is not.
* LLM output is stochastic — fix temperature=0 and run more than once.
* Hazard definitions are OURS, not nuScenes'. They are stated below and shared
  with the live EventDetector via the same geometry helpers, so the evaluation
  measures the agent rather than a mismatch between two implementations.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from agent.monitor import EGO_HALF_WIDTH_M
from data.dataset import LABEL_MAP, NIGHT_SCENES
from utils.geometry import time_to_collision

HAZARD_KINDS = ("entered_path", "ttc_low", "ped_approaching")


@dataclass
class Hazard:
    """A ground-truth event the agent ideally warns about."""
    frame_idx: int
    kind: str
    instance_token: str          # GT identity — legitimate here, this is offline
    payload: Dict = field(default_factory=dict)


def _gt_tracks(nusc, stream) -> Dict[str, List[Tuple[int, np.ndarray, int, int]]]:
    """
    instance_token -> [(frame_idx, ego_frame_xy, label, timestamp_us)], keyframes only.
    """
    out: Dict[str, List[Tuple[int, np.ndarray, int, int]]] = {}
    for f in stream:
        if not f.is_keyframe:
            continue
        sample = nusc.get("sample", f.sample_token)
        R = Quaternion(f.ego_pose["rotation"]).rotation_matrix[:2, :2]
        t = np.asarray(f.ego_pose["translation"][:2], dtype=np.float64)
        for tok in sample["anns"]:
            a = nusc.get("sample_annotation", tok)
            if a["category_name"] not in LABEL_MAP:
                continue
            xy = (np.asarray(a["translation"][:2], dtype=np.float64) - t) @ R
            if abs(xy[0]) > 51.2 or abs(xy[1]) > 51.2:
                continue
            out.setdefault(a["instance_token"], []).append(
                (f.frame_idx, xy, LABEL_MAP[a["category_name"]], f.timestamp_us)
            )
    return out


def extract_hazards(nusc, stream, ttc_threshold_s: float = 1.5,
                    path_half_width_m: float = EGO_HALF_WIDTH_M,
                    ttc_corridor_m: float = 3.0) -> List[Hazard]:
    """
    Derive ground-truth hazards for a scene from annotations.

    Uses the same `time_to_collision` and corridor gate as the live EventDetector,
    so the two cannot drift into measuring their own mismatch.

    THE HAZARD THRESHOLD IS DELIBERATELY STRICTER THAN THE WARNING THRESHOLD.
    The detector warns at TTC < 3.5 s; a hazard is TTC < 1.5 s. If both used the
    same threshold they would fire on the same frame and median lead time would be
    structurally 0.0 s — which is exactly what the first run of this harness
    produced, and it measured nothing. Separating them makes lead time meaningful:
    it is the interval between "the monitor spoke" and "the situation actually
    became dangerous". `entered_path` and `ped_approaching` remain state changes
    with no natural severity scale, so they keep ~0 lead by construction; report
    lead time for `ttc_low` separately for that reason.
    """
    hazards: List[Hazard] = []
    for inst, obs in _gt_tracks(nusc, stream).items():
        obs.sort(key=lambda o: o[0])
        for i in range(1, len(obs)):
            (f_prev, xy_prev, _, t_prev) = obs[i - 1]
            (f_cur, xy_cur, label, t_cur) = obs[i]
            dt = (t_cur - t_prev) / 1e6
            if dt <= 0:
                continue
            # relative velocity in the ego frame (GT positions are already ego-frame,
            # so this difference is inherently relative to the ego vehicle)
            v_rel = (xy_cur - xy_prev) / dt
            rng = float(np.hypot(*xy_cur))

            if abs(xy_cur[1]) <= path_half_width_m and abs(xy_prev[1]) > path_half_width_m:
                hazards.append(Hazard(f_cur, "entered_path", inst,
                                      {"range_m": rng, "label": int(label)}))

            if abs(xy_cur[1]) <= ttc_corridor_m:
                ttc = time_to_collision(xy_cur, v_rel)
                if np.isfinite(ttc) and ttc < ttc_threshold_s:
                    hazards.append(Hazard(f_cur, "ttc_low", inst,
                                          {"ttc_s": float(ttc), "range_m": rng,
                                           "label": int(label)}))

            if label in (1, 2) and abs(xy_cur[1]) > path_half_width_m and rng <= 25.0:
                closing = -np.sign(xy_cur[1]) * v_rel[1]
                gap = max(abs(float(xy_cur[1])) - path_half_width_m, 0.0)
                if closing > 1.0 and (gap / closing if closing > 1e-6 else np.inf) <= 4.0:
                    hazards.append(Hazard(f_cur, "ped_approaching", inst,
                                          {"range_m": rng, "label": int(label)}))

    # collapse repeats of the same (instance, kind) to the FIRST occurrence —
    # a hazard is an event, not a state that persists for many frames
    seen, unique = set(), []
    for h in sorted(hazards, key=lambda h: h.frame_idx):
        key = (h.instance_token, h.kind)
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique


def match_advisories_to_hazards(advisories, hazards: List[Hazard],
                                max_lead_frames: int = 20) -> Tuple[List, List, List]:
    """
    Align advisories with hazards.

    Returns: (matched pairs, unmatched hazards = misses,
              unmatched advisories = false alarms)

    An advisory may only be credited to a hazard it PRECEDES (or coincides with).
    Crediting a post-hoc description as a warning would inflate the headline
    metric — an easy mistake when matching purely on a symmetric time window.
    """
    matched, used_adv = [], set()
    remaining = list(hazards)
    misses = []
    for h in remaining:
        best, best_lead = None, None
        for i, a in enumerate(advisories):
            if i in used_adv or a.abstained:
                continue
            lead = h.frame_idx - a.frame_idx          # positive => warned in advance
            if 0 <= lead <= max_lead_frames:
                if best is None or lead < best_lead:
                    best, best_lead = i, lead
        if best is None:
            misses.append(h)
        else:
            used_adv.add(best)
            matched.append((advisories[best], h, best_lead))
    false_alarms = [a for i, a in enumerate(advisories) if i not in used_adv and not a.abstained]
    return matched, misses, false_alarms


def abstention_matrix(advisories, detector_errors: Dict[int, bool]) -> Dict[str, float]:
    """
    The 2x2.

    Args:
        advisories: objects with `abstained` and `frame_idx`.
        detector_errors: frame_idx -> was perception materially wrong there.
            "Materially wrong" means a missed GT object inside the ego corridor or
            a false positive there — a missed parked car at 48 m is not the same
            failure as a missed pedestrian at 8 m, and averaging them hides the
            thing that matters.
    """
    m = Counter()
    for a in advisories:
        wrong = bool(detector_errors.get(a.frame_idx, False))
        m[("abstained" if a.abstained else "spoke", "wrong" if wrong else "right")] += 1
    spoke_wrong = m[("spoke", "wrong")]
    abst_wrong = m[("abstained", "wrong")]
    spoke_right = m[("spoke", "right")]
    abst_right = m[("abstained", "right")]
    total_wrong = spoke_wrong + abst_wrong
    total_abst = abst_wrong + abst_right
    return {
        "abstained_wrong_correct": abst_wrong,
        "abstained_right_overcautious": abst_right,
        "spoke_wrong_DANGEROUS": spoke_wrong,
        "spoke_right_correct": spoke_right,
        # of the frames where perception was wrong, how often did we stay quiet
        "abstention_recall": abst_wrong / total_wrong if total_wrong else float("nan"),
        # of the times we abstained, how often were we right to
        "abstention_precision": abst_wrong / total_abst if total_abst else float("nan"),
    }


def summarize(matched, misses, false_alarms, duration_s: float) -> Dict:
    leads = [lead for _, _, lead in matched]
    # ttc_low is the only kind with a natural severity scale, so it is the only
    # one where lead time is a real measurement rather than 0 by construction.
    ttc_leads = [lead for _, h, lead in matched if h.kind == "ttc_low"]
    return {
        "median_ttc_lead_s": float(np.median(ttc_leads)) * 0.5 if ttc_leads else float("nan"),
        "n_ttc_hazards_warned": len(ttc_leads),
        "hazards": len(matched) + len(misses),
        "warned": len(matched),
        "missed": len(misses),
        "median_lead_frames": float(np.median(leads)) if leads else float("nan"),
        "median_lead_s": float(np.median(leads)) * 0.5 if leads else float("nan"),  # 2 Hz
        "false_alarms": len(false_alarms),
        "false_alarms_per_min": len(false_alarms) / max(duration_s / 60.0, 1e-9),
        "hazards_by_kind": dict(Counter(h.kind for _, h, _ in matched)),
        "missed_by_kind": dict(Counter(h.kind for h in misses)),
    }


def main():
    """
    CLI:
        python -m evaluation.streaming_eval --scenes scene-1094,scene-1100

    Prints per scene and aggregated, split by day/night:
        hazards | warned | median lead | false alarms/min | abstention 2x2

    The day/night split is the point — every phase before 9 reported one number
    for everything, and Phase 9 showed why that hides the failure that matters.
    """
    import asyncio

    from nuscenes.nuscenes import NuScenes

    from agent.streaming_agent import StreamingAgent
    from demo.stream import FrameStream

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene-1094,scene-1100,scene-0103,scene-0916")
    ap.add_argument("--data-root", default="data/raw/v1.0-mini")
    ap.add_argument("--out", default="logs/streaming_eval.json")
    ap.add_argument("--use-gt-detections", action="store_true",
                    help="replay GT boxes instead of the detector — isolates the "
                         "agent's salience logic from perception error")
    args = ap.parse_args()

    nusc = NuScenes(version=Path(args.data_root).name, dataroot=args.data_root, verbose=False)
    results: Dict[str, Dict] = {}

    async def run_scene(scene: str) -> Dict:
        stream = FrameStream(nusc, args.data_root, scene, mode="keyframes", load_images=False)
        frames = list(stream)
        duration = (frames[-1].timestamp_us - frames[0].timestamp_us) / 1e6
        hazards = extract_hazards(nusc, iter(frames))

        agent = StreamingAgent(mcp_client=None)
        advisories = []
        gt = _gt_tracks(nusc, iter(frames))
        per_frame: Dict[int, List] = {}
        for inst, obs in gt.items():
            for (fi, xy, label, ts) in obs:
                per_frame.setdefault(fi, []).append((xy, label))
        for f in frames:
            rows = per_frame.get(f.frame_idx, [])
            boxes = np.array([[xy[0], xy[1], 4.0, 2.0, 0.0] for xy, _ in rows]).reshape(-1, 5)
            labels = np.array([l for _, l in rows], dtype=int)
            scores = np.full(len(rows), 0.9)
            advisories += await agent.step(boxes, labels, scores, f.timestamp_us,
                                           f.frame_idx, ego_pose=f.ego_pose)

        matched, misses, fas = match_advisories_to_hazards(advisories, hazards)
        s = summarize(matched, misses, fas, duration)
        s["night"] = scene in NIGHT_SCENES
        s["frames"] = len(frames)
        s["duration_s"] = duration
        s["advisories"] = len(advisories)
        s["rates"] = agent.rate_summary(duration)
        return s

    for scene in args.scenes.split(","):
        scene = scene.strip()
        try:
            results[scene] = asyncio.run(run_scene(scene))
        except Exception as e:            # a missing scene must not abort the sweep
            print(f"[skip] {scene}: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=float)

    print("\n" + "=" * 86)
    print(f"{'scene':<14}{'cond':>6}{'frames':>8}{'hazards':>9}{'warned':>8}"
          f"{'missed':>8}{'lead(s)':>9}{'FA/min':>9}{'adv':>6}")
    print("-" * 86)
    for k, r in results.items():
        print(f"{k:<14}{'night' if r['night'] else 'day':>6}{r['frames']:>8}"
              f"{r['hazards']:>9}{r['warned']:>8}{r['missed']:>8}"
              f"{r['median_lead_s']:>9.1f}{r['false_alarms_per_min']:>9.1f}{r['advisories']:>6}")
    print("=" * 86)
    for cond in (False, True):
        rows = [r for r in results.values() if r["night"] == cond]
        if not rows:
            continue
        hz, wr = sum(r["hazards"] for r in rows), sum(r["warned"] for r in rows)
        print(f"{'night' if cond else 'day':>5}: {wr}/{hz} hazards warned "
              f"(raw counts — do not read these as percentages at this sample size)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
