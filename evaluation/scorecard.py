"""
P14-2 — The unified scorecard.

One command, one row. Every experiment from Phase 14 on is scored by this file and
nothing else, so rows are comparable by construction (operating rule 1). It runs
the three evaluations the project already trusts and flattens them into one set
of cells:

    Phase 9  day/night audit       det.{seen_day, unseen_day, unseen_night, unseen_miniday}.mAP
                                   det.night_gap_rel = (unseen_day - unseen_night) / unseen_day
    Phase 13 foreign cameras       det.foreign.{phone, dashcam, action_cam, mean}.mAP
    Phase 10 range-bucketed BEV    bev.{unseen_day, unseen_night}.mAP
                                   bev.{unseen_day, unseen_night}.{near, mid, far}.mAP

`seen_day` is the fit ceiling (the gameplan's "native day"). The foreign-camera
benchmark's undegraded "native" control is not re-run: it is the same frames and
the same transform as `unseen_day`, and the logs agree to 1e-16.

Foreign cameras are scored WITH FOV normalisation by default, because that is the
path the BYO demo actually runs (Phase 13's lesson: benchmark the pipeline you
deploy). `--foreign-raw` scores the un-normalised worst case instead, and the flag
is recorded in the row so the two are never mixed silently.

SEEDS (rule 2)
--------------
Every cell reports mean, sample std, a t-based 95% interval and n. Pass a
checkpoint path containing "{seed}" plus `--seeds 0,1,2` to score one checkpoint
per seed. A path without "{seed}" is scored once and reported as n=1, with a null
interval: one run is a data point, not a distribution. Derived cells (the night
gap, the foreign mean) are computed per seed and then aggregated, so the day and
night numbers that form a gap always come from the same checkpoint.

OUTPUT
------
    results/scorecard/<name>.json   the full row: per-seed values, per-class AP,
                                    frame and GT counts, scenes, git SHA, config
                                    hashes, wall time
    results/scorecard.csv           the leaderboard: one row per experiment, the
                                    mean and 95% half-width of every cell. Scoring
                                    an existing name replaces its row.

Usage (the defaults are the current baselines):
    python -m evaluation.scorecard --name baseline
    python -m evaluation.scorecard --name r1_dinov2 \\
        --det-config configs/detector_dinov2.yaml \\
        --det-ckpt 'checkpoints/detector_dinov2_s{seed}.pt' --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

AUDIT_CELLS = ("seen_day", "unseen_day", "unseen_night", "unseen_miniday")
FOREIGN_CAMERAS = ("phone", "dashcam", "action_cam")
BEV_CONDITIONS = ("unseen_day", "unseen_night")
# Must match evaluation.radar_ablation.RANGE_BUCKETS (a test enforces it). Not
# imported, so this module's pure helpers load without torch or nuScenes.
RANGE_NAMES = ("near", "mid", "far")

CELL_ORDER: Tuple[str, ...] = (
    "det.seen_day.mAP",
    "det.unseen_day.mAP",
    "det.unseen_night.mAP",
    "det.unseen_miniday.mAP",
    "det.night_gap_rel",
    "det.foreign.phone.mAP",
    "det.foreign.dashcam.mAP",
    "det.foreign.action_cam.mAP",
    "det.foreign.mean.mAP",
    "bev.unseen_day.mAP",
    "bev.unseen_day.near.mAP",
    "bev.unseen_day.mid.mAP",
    "bev.unseen_day.far.mAP",
    "bev.unseen_night.mAP",
    "bev.unseen_night.near.mAP",
    "bev.unseen_night.mid.mAP",
    "bev.unseen_night.far.mAP",
)

META_COLUMNS = (
    "experiment", "created_at", "git_sha", "git_dirty",
    "det_n", "det_config", "det_config_hash",
    "bev_n", "bev_config", "bev_config_hash",
    "foreign_fov_normalize", "notes",
)
CI_SUFFIX = ".ci95"
MIN_SEEDS = 3
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# ── seeds and checkpoints ───────────────────────────────────────────────────

def parse_seeds(text: Optional[str]) -> List[int]:
    """'0,1,2' -> [0, 1, 2]; None or '' -> []."""
    if not text:
        return []
    seeds = [int(s) for s in text.split(",") if s.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {text}")
    return seeds


def validate_seed_args(seeds: Sequence[int], templates: Sequence[str]) -> None:
    """
    Refuse `--seeds` when no checkpoint path contains "{seed}".

    Otherwise one checkpoint would be scored N times and reported as n=N with a
    zero-width interval, which is exactly the false confidence rule 2 exists to
    prevent.
    """
    if seeds and not any("{seed}" in t for t in templates):
        raise ValueError("--seeds was given but no checkpoint path contains '{seed}'; "
                         "one checkpoint scored N times is still n=1")


def expand_ckpts(template: str, seeds: Sequence[int]) -> Tuple[List[str], List[Optional[int]]]:
    """
    Checkpoint paths and their seeds. A template with "{seed}" expands once per
    seed; one without it is a single checkpoint whose seed is unknown (None).
    """
    if seeds and "{seed}" in template:
        return [template.format(seed=s) for s in seeds], list(seeds)
    if "{seed}" in template:
        raise ValueError(f"'{template}' contains '{{seed}}' but no --seeds were given")
    return [template], [None]


# ── flattening evaluator output into cells ──────────────────────────────────

def detector_seed_cells(audit: Dict[str, dict], foreign: Dict[str, dict]) -> Tuple[Dict[str, float], dict]:
    """
    One detector checkpoint's cells.

    Args: audit — `evaluate_audit_cells` output (cell -> {mAP, AP, num_frames, ...});
      foreign — `evaluate_foreign` output (camera -> {mAP, AP, frames, ...}).
    Returns: (cells, details) — flat cell -> value, plus per-class AP and frame
    counts for the JSON row.

    The night gap and the foreign mean are derived HERE, per seed, so each is
    built from numbers produced by one and the same checkpoint.
    """
    cells: Dict[str, float] = {}
    details: dict = {"AP": {}, "frames": {}}
    for name in AUDIT_CELLS:
        if name in audit:
            cells[f"det.{name}.mAP"] = float(audit[name]["mAP"])
            details["AP"][f"det.{name}"] = list(audit[name].get("AP", []))
            details["frames"][f"det.{name}"] = audit[name].get("num_frames")
    day, night = cells.get("det.unseen_day.mAP"), cells.get("det.unseen_night.mAP")
    if day is not None and night is not None and day > 0:
        cells["det.night_gap_rel"] = (day - night) / day
    for cam in FOREIGN_CAMERAS:
        if cam in foreign:
            cells[f"det.foreign.{cam}.mAP"] = float(foreign[cam]["mAP"])
            details["AP"][f"det.foreign.{cam}"] = list(foreign[cam].get("AP", []))
            details["frames"][f"det.foreign.{cam}"] = foreign[cam].get("frames")
    cams = [cells.get(f"det.foreign.{c}.mAP") for c in FOREIGN_CAMERAS]
    if all(v is not None for v in cams):
        cells["det.foreign.mean.mAP"] = sum(cams) / len(cams)
    return cells, details


def bev_seed_cells(bev: Dict[str, dict]) -> Tuple[Dict[str, float], dict]:
    """
    One BEV checkpoint's cells.

    Args: bev — condition -> `radar_ablation.evaluate_cell` output
      ({mAP, AP, num_frames, num_gt, buckets: {near/mid/far: {mAP, AP, num_gt}}}).
    Returns: (cells, details). GT counts go in details because an AP over a
    handful of boxes is not a measurement and the far/night bucket is thin.
    """
    cells: Dict[str, float] = {}
    details: dict = {"AP": {}, "num_gt": {}, "frames": {}, "mean_gate": {}}
    for cond in BEV_CONDITIONS:
        if cond not in bev:
            continue
        r = bev[cond]
        cells[f"bev.{cond}.mAP"] = float(r["mAP"])
        details["AP"][f"bev.{cond}"] = list(r.get("AP", []))
        details["num_gt"][f"bev.{cond}"] = r.get("num_gt")
        details["frames"][f"bev.{cond}"] = r.get("num_frames")
        if "mean_gate" in r:
            details["mean_gate"][f"bev.{cond}"] = r["mean_gate"]
        for rng in RANGE_NAMES:
            b = r.get("buckets", {}).get(rng)
            if b is None:
                continue
            cells[f"bev.{cond}.{rng}.mAP"] = float(b["mAP"])
            details["AP"][f"bev.{cond}.{rng}"] = list(b.get("AP", []))
            details["num_gt"][f"bev.{cond}.{rng}"] = b.get("num_gt")
    return cells, details


# ── statistics ──────────────────────────────────────────────────────────────

def summarize(values: Sequence[Optional[float]]) -> dict:
    """
    {mean, std, ci95: [lo, hi], n, values} over the non-missing per-seed values.

    The interval is mean +/- t(0.975, n-1) * s / sqrt(n) with the sample std s.
    With three seeds t = 4.30, so the interval is wide; that is the honest width
    of three runs, and a gain inside it is not a gain. n=1 gives null std and
    interval rather than a fake zero-width one.
    """
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(vals)
    out = {"mean": None, "std": None, "ci95": None, "n": n,
           "values": [None if v is None else float(v) for v in values]}
    if n == 0:
        return out
    mean = sum(vals) / n
    out["mean"] = mean
    if n >= 2:
        from scipy.stats import t
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        half = float(t.ppf(0.975, n - 1)) * std / math.sqrt(n)
        out["std"] = std
        out["ci95"] = [mean - half, mean + half]
    return out


def aggregate(per_seed: Sequence[Dict[str, float]]) -> Dict[str, dict]:
    """cell -> summarize() across seeds, for every cell any seed reported."""
    keys = ordered_cells(k for d in per_seed for k in d)
    return {k: summarize([d.get(k) for d in per_seed]) for k in keys}


def ordered_cells(keys) -> List[str]:
    """CELL_ORDER first, then any cell it does not know, sorted."""
    keys = set(keys)
    return [k for k in CELL_ORDER if k in keys] + sorted(keys - set(CELL_ORDER))


# ── the row ─────────────────────────────────────────────────────────────────

def build_row(name: str, det_per_seed: Sequence[Dict[str, float]],
              bev_per_seed: Sequence[Dict[str, float]], *, components: dict,
              data: dict, settings: dict, details: dict, wall_time_s: float,
              notes: str = "", git: Optional[dict] = None) -> dict:
    """
    Assemble one scorecard row. Every CELL_ORDER cell is present, as n=0 with a
    null mean if nothing produced it, so a skipped evaluation shows up as a gap
    rather than as a missing column.
    """
    from utils.provenance import git_info

    cells = {**aggregate(det_per_seed), **aggregate(bev_per_seed)}
    for k in CELL_ORDER:
        cells.setdefault(k, summarize([]))
    cells = {k: cells[k] for k in ordered_cells(cells)}
    return {
        "schema": SCHEMA_VERSION,
        "experiment": name,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(float(wall_time_s), 1),
        "git": git if git is not None else git_info(),
        "components": components,
        "settings": settings,
        "data": data,
        "cells": cells,
        "details": details,
    }


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def csv_record(row: dict) -> Dict[str, str]:
    """Flatten a row into one leaderboard record: metadata, then mean and 95% half-width per cell."""
    comp = row.get("components", {})
    det, bev = comp.get("detector") or {}, comp.get("bev") or {}
    git = row.get("git", {})
    rec = {
        "experiment": row["experiment"],
        "created_at": row.get("created_at", ""),
        "git_sha": (git.get("sha") or "")[:12],
        "git_dirty": _fmt(git.get("dirty")),
        "det_n": _fmt(len(det.get("ckpts", [])) or None),
        "det_config": det.get("config", ""),
        "det_config_hash": det.get("config_hash", ""),
        "bev_n": _fmt(len(bev.get("ckpts", [])) or None),
        "bev_config": bev.get("config", ""),
        "bev_config_hash": bev.get("config_hash", ""),
        "foreign_fov_normalize": _fmt(row.get("settings", {}).get("foreign_fov_normalize")),
        "notes": row.get("notes", ""),
    }
    for cell, s in row["cells"].items():
        rec[cell] = _fmt(s["mean"])
        ci = s.get("ci95")
        rec[cell + CI_SUFFIX] = _fmt((ci[1] - ci[0]) / 2) if ci else ""
    return rec


def upsert_csv(path, row: dict) -> List[str]:
    """
    Write `row` into the leaderboard at `path`, replacing any row with the same
    experiment name, and return the header.

    Columns are the union over all rows, in a fixed order (metadata, then
    CELL_ORDER, then unknown cells sorted), each cell followed by its half-width,
    so adding a cell later never reorders the existing ones.
    """
    path = Path(path)
    new = csv_record(row)
    records: List[Dict[str, str]] = []
    if path.exists():
        with path.open(newline="") as f:
            records = [r for r in csv.DictReader(f) if r.get("experiment") != row["experiment"]]
    records.append(new)
    cells = ordered_cells(k for r in records for k in r
                          if k not in META_COLUMNS and not k.endswith(CI_SUFFIX))
    header = list(META_COLUMNS) + [c for cell in cells for c in (cell, cell + CI_SUFFIX)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in header})
    return header


def format_summary(row: dict) -> str:
    """Terminal table: cell, mean, 95% interval, n."""
    lines = ["=" * 64, f"{'cell':<30}{'mean':>9}{'95% interval':>18}{'n':>5}", "-" * 64]
    for cell, s in row["cells"].items():
        mean = "-" if s["mean"] is None else f"{s['mean']:.4f}"
        ci = "" if not s["ci95"] else f"[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"
        lines.append(f"{cell:<30}{mean:>9}{ci:>18}{s['n']:>5}")
    lines.append("=" * 64)
    return "\n".join(lines)


# ── running the evaluations ─────────────────────────────────────────────────

def resolve_ckpts(args) -> Tuple[List[str], List[Optional[int]], List[str], List[Optional[int]]]:
    """(det_ckpts, det_seeds, bev_ckpts, bev_seeds) from the CLI; ValueError on a bad combination."""
    seeds = parse_seeds(args.seeds)
    use_bev = not args.no_bev
    validate_seed_args(seeds, [args.det_ckpt] + ([args.bev_ckpt] if use_bev else []))
    det_ckpts, det_seeds = expand_ckpts(args.det_ckpt, seeds)
    bev_ckpts, bev_seeds = expand_ckpts(args.bev_ckpt, seeds) if use_bev else ([], [])
    return det_ckpts, det_seeds, bev_ckpts, bev_seeds


def run_scorecard(args, det_ckpts, det_seeds, bev_ckpts, bev_seeds) -> dict:
    """Score every checkpoint on every cell and return the row. Needs data and checkpoints."""
    import yaml
    from nuscenes.nuscenes import NuScenes

    from data.dataset import version_from_data_root
    from evaluation.day_night_audit import audit_cells, evaluate_audit_cells
    from evaluation.foreign_camera_eval import evaluate_foreign
    from evaluation.radar_ablation import build_loader, evaluate_cell, load_model
    from models.detection.train_detector import _pick_device, load_detector
    from utils.provenance import config_hash

    t0 = time.time()
    use_bev = bool(bev_ckpts)

    # Fail before loading nuScenes (slow) if anything is missing.
    missing = [c for c in det_ckpts + bev_ckpts if not Path(c).exists()]
    if missing:
        raise SystemExit("checkpoint(s) not found:\n  " + "\n  ".join(missing))
    for label, n in (("detector", len(det_ckpts)), ("BEV", len(bev_ckpts))):
        if 0 < n < MIN_SEEDS:
            print(f"[warn] {label}: {n} checkpoint(s). Rule 2 asks for >= {MIN_SEEDS} seeds; "
                  f"its cells will carry no usable interval.")

    device = _pick_device()
    tv_root, mini_root = Path(args.trainval_root), Path(args.mini_root)
    nusc_tv = NuScenes(version=version_from_data_root(tv_root), dataroot=str(tv_root), verbose=False)
    nusc_mini = NuScenes(version=version_from_data_root(mini_root), dataroot=str(mini_root), verbose=False)
    cells, mini_cond = audit_cells(nusc_tv, tv_root, nusc_mini, mini_root)
    scenes = {name: sets for name, _, _, sets in cells}
    fov_norm = not args.foreign_raw

    det_cfg = yaml.safe_load(open(args.det_config))
    det_per_seed, det_details = [], []
    for ckpt, seed in zip(det_ckpts, det_seeds):
        print(f"\n##### detector {ckpt} (seed {seed})")
        model = load_detector(det_cfg, ckpt, device)
        audit = evaluate_audit_cells(model, det_cfg, device, cells)
        foreign = evaluate_foreign(model, det_cfg, device, nusc_tv, tv_root,
                                   scenes["unseen_day"], FOREIGN_CAMERAS,
                                   fov_normalize=fov_norm, include_native=False)
        c, d = detector_seed_cells(audit, foreign)
        det_per_seed.append(c)
        det_details.append({"seed": seed, "ckpt": ckpt, **d})
        del model

    bev_cfg, bev_per_seed, bev_details = None, [], []
    if use_bev:
        bev_cfg = yaml.safe_load(open(args.bev_config))
        use_radar = bool(bev_cfg.get("use_radar", False))
        conditions = [("unseen_day", nusc_tv, tv_root, scenes["unseen_day"]),
                      ("unseen_night", nusc_mini, mini_root, scenes["unseen_night"])]
        for ckpt, seed in zip(bev_ckpts, bev_seeds):
            print(f"\n##### BEV {ckpt} (seed {seed})")
            model = load_model(bev_cfg, ckpt, device)
            res = {}
            for cname, nusc, root, sc in conditions:
                loader = build_loader(bev_cfg, nusc, root, sc, use_radar)
                print(f"[bev / {cname}] {len(sc)} scenes, {len(loader.dataset)} frames")
                res[cname] = evaluate_cell(model, loader, device, bev_cfg["num_classes"],
                                           tuple(bev_cfg["xbound"]), tuple(bev_cfg["ybound"]))
            c, d = bev_seed_cells(res)
            bev_per_seed.append(c)
            bev_details.append({"seed": seed, "ckpt": ckpt, **d})
            del model

    components = {
        "detector": {"config": args.det_config, "config_hash": config_hash(det_cfg),
                     "ckpts": det_ckpts, "seeds": det_seeds},
        "bev": None if bev_cfg is None else {
            "config": args.bev_config, "config_hash": config_hash(bev_cfg),
            "ckpts": bev_ckpts, "seeds": bev_seeds},
    }
    data = {
        "trainval_root": str(tv_root), "mini_root": str(mini_root),
        "scenes": {k: sorted(v) for k, v in scenes.items()},
        "mini_conditions": mini_cond,
    }
    settings = {"foreign_fov_normalize": fov_norm, "foreign_cameras": list(FOREIGN_CAMERAS),
                "device": str(device)}
    return build_row(args.name, det_per_seed, bev_per_seed, components=components,
                     data=data, settings=settings,
                     details={"detector": det_details, "bev": bev_details},
                     wall_time_s=time.time() - t0, notes=args.notes)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m evaluation.scorecard",
        description="Score a detector (and optionally a BEV model) on every scorecard cell.")
    ap.add_argument("--name", required=True,
                    help="experiment name; the JSON file name and the leaderboard key")
    ap.add_argument("--det-config", default="configs/detector.yaml")
    ap.add_argument("--det-ckpt", default="checkpoints/detector_best.pt",
                    help="checkpoint path; may contain {seed}")
    ap.add_argument("--bev-config", default="configs/bev_surround_p10.yaml")
    ap.add_argument("--bev-ckpt", default="checkpoints/bev_surround_p10_last.pt",
                    help="checkpoint path; may contain {seed}")
    ap.add_argument("--seeds", default="", help="comma-separated, e.g. 0,1,2; expands {seed}")
    ap.add_argument("--no-bev", action="store_true", help="skip the BEV cells (they stay empty)")
    ap.add_argument("--foreign-raw", action="store_true",
                    help="score foreign cameras WITHOUT the deployed FOV normalisation")
    ap.add_argument("--trainval-root", default="data/raw/v1.0-trainval")
    ap.add_argument("--mini-root", default="data/raw/v1.0-mini")
    ap.add_argument("--out-dir", default="results/scorecard")
    ap.add_argument("--csv", default="results/scorecard.csv")
    ap.add_argument("--notes", default="")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not _NAME_RE.match(args.name):
        raise SystemExit(f"--name must match {_NAME_RE.pattern}: {args.name!r}")
    try:
        ckpts = resolve_ckpts(args)
    except ValueError as e:
        raise SystemExit(str(e))
    row = run_scorecard(args, *ckpts)

    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(row, open(out, "w"), indent=2)
    upsert_csv(args.csv, row)
    print("\n" + format_summary(row))
    print(f"wrote {out}\nupdated {args.csv} (row '{args.name}')")


if __name__ == "__main__":
    main()
