"""
P12-5 — Evaluating a streaming agent.

This is the part that turns the demo into a result. A video of an agent saying
sensible things is a demo; a table of lead times, false-alarm rates and
abstention correctness is evidence. Without this file Phase 12 is a toy.

Run on KEYFRAMES only (2 Hz) — sweeps carry no annotations, so nothing said on a
sweep frame is checkable. The demo runs at 12 Hz; the metrics run at 2 Hz.

FOUR METRICS
------------
1. WARNING LEAD TIME
   For each ground-truth hazard, how long before it materialised did the agent
   warn? Define a hazard from GT: a GT object entering the ego corridor, or GT
   TTC dropping below threshold. Lead time = t_hazard - t_advisory.
   Report the median and the fraction warned at all. Negative lead time (warned
   after the fact) is a failure, not a small delay — count it separately rather
   than letting it drag the mean down and disappear.

2. FALSE-ALARM RATE
   Advisories per minute with no corresponding GT hazard. This is the metric that
   punishes a chatty agent, and it is why the fast tier's hysteresis matters.
   An agent that warns constantly trivially achieves perfect lead time; report
   these two together or neither is meaningful.

3. ABSTENTION CORRECTNESS
   The Phase 11/12 payoff. Build the 2x2:

                        perception was WRONG   perception was RIGHT
       abstained              correct               over-cautious
       spoke confidently   DANGEROUS FAILURE            correct

   The bottom-left cell is the one that matters: confident advisories issued on
   frames where the detector was wrong. Phase 9 says these concentrate at night.
   The headline claim to test is that abstention shrinks that cell on night
   scenes without emptying the top-right one on day scenes — i.e. the system
   gets appropriately quiet in the dark, not uniformly useless.

4. GROUNDING ACCURACY
   Of the numeric claims in advisories (ranges, counts, TTC), how many match GT
   within tolerance? Reuse the approach in `evaluation/agent_eval/` — it already
   builds GT-derived questions and scores answers, so extend rather than
   duplicate it. Parse numbers out of the advisory text and compare against the
   frame's GT.

METHODOLOGICAL WARNINGS
-----------------------
* Sample size is small. Three night scenes, ~121 keyframes, and hazards are rarer
  still — you may have a few dozen hazard events total. Report raw counts
  everywhere and resist percentages computed over single-digit denominators.
  "2 of 3 hazards warned" is honest; "67% hazard recall" is not.
* LLM output is stochastic. Fix `temperature=0`, and run the full evaluation more
  than once to gauge variance. Report the spread.
* Cost is real. A full sweep over all scenes with an LLM in the loop costs money
  and time; cache advisories keyed by (scene, frame_idx, event hash) so reruns
  of the *metrics* do not re-invoke the model.
* Hazard definitions are yours, not nuScenes'. Write them down precisely in the
  README. A reader must be able to tell whether "hazard" was defined before or
  after seeing the results, and the only way to establish that is to state the
  definition plainly and keep it fixed across day and night.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Hazard:
    """A ground-truth event the agent ideally warns about."""
    frame_idx: int
    kind: str                  # "entered_path" | "ttc_low" | "ped_approaching"
    instance_token: str        # GT identity — fine here, this is offline eval
    payload: Dict


def extract_hazards(nusc, scene_name: str, ttc_threshold_s: float = 4.0,
                    path_half_width_m: float = 2.0) -> List[Hazard]:
    """
    Derive ground-truth hazards for a scene from annotations.

    Mirror EventDetector's rules exactly, but compute them from GT boxes and GT
    ego poses instead of predictions. If the two rule sets drift apart, the
    evaluation measures the mismatch between your two implementations rather than
    the agent's performance — factor the geometry into a shared helper that both
    the detector and this function call.
    """
    raise NotImplementedError("P12-5")


def match_advisories_to_hazards(advisories, hazards, max_lead_frames: int = 20
                                ) -> Tuple[List[Tuple], List, List]:
    """
    Align advisories with hazards.

    Returns: (matched pairs, unmatched hazards = misses,
              unmatched advisories = false alarms)

    Match on object identity where possible (the advisory's track, associated to
    a GT instance via centre distance) and otherwise on time proximity. An
    advisory may only be credited to a hazard it PRECEDES — crediting a
    post-hoc description as a warning would inflate the headline metric, and it
    is an easy mistake to make when matching purely on time windows.
    """
    raise NotImplementedError("P12-5")


def abstention_matrix(advisories, detector_errors) -> Dict[str, int]:
    """
    The 2x2 above.

    Args:
        advisories: with `abstained` and `frame_idx`.
        detector_errors: per-frame flag for whether perception was materially
            wrong (e.g. a missed GT object inside the ego corridor, or a false
            positive there). Define "materially wrong" in the README — a missed
            parked car at 48 m is not the same failure as a missed pedestrian at
            8 m, and averaging them together hides the thing you care about.

    Returns counts for all four cells plus the derived rates.
    """
    raise NotImplementedError("P12-5")


def main():
    """
    CLI:
        python -m evaluation.streaming_eval --scenes scene-1094,scene-1100 \
            --config configs/demo.yaml

    Print, per scene and aggregated, split by day/night:
        hazards | warned | median lead time | false alarms/min | abstention 2x2

    The day/night split is the point. Every earlier phase reported one number for
    everything; Phase 9 showed why that hides the failure that matters.
    """
    raise NotImplementedError("P12-5")


if __name__ == "__main__":
    main()
