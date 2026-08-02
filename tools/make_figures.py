"""
Generate the Phase 9–11 result figures from the JSON already on disk.

    python -m tools.make_figures            # writes docs/figures/*.png (light + dark)

Sources: logs/day_night_audit.json, logs/radar_ablation.json,
         logs/introspection_nomc.json, logs/introspection_mc.json

DESIGN NOTES (why the charts look the way they do)
--------------------------------------------------
* Two series only — "camera" and "camera + radar", or "head" and "raw score" —
  so the categorical palette uses slots 1 and 2 (blue / orange). That pair was
  validated with the palette checker in both modes: CVD ΔE 24.7 light / 26.8
  dark against a ≥8 target, normal-vision ΔE 33.6 / 31.8 against a ≥15 floor.
* One y-axis per chart, never two. Where two measures share a panel they share
  units (mAP), so no second scale is needed.
* Bars capped at a fixed thickness with the slot's leftover left as air; 2px
  surface-coloured gaps separate adjacent bars rather than strokes.
* Labels are selective — the story values carry a number, the rest are read off
  the axis. Text uses ink tokens, never the series colour; identity comes from
  the legend swatch beside it.
* Both light and dark variants are emitted from the same code, with the dark
  steps taken from the palette's dark column rather than by inverting the light
  ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/figures")

# Validated categorical slots 1 and 2, plus surfaces and ink, per the palette.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  grid="#e4e3df", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                  grid="#33322f", s1="#3987e5", s2="#d95926", s3="#199e70"),
}

BAR_W = 0.26          # leaves air in each slot rather than filling it (<=24px)
GAP = 0.02            # surface-coloured separation between adjacent bars


def _style(ax, t, title, ylabel, subtitle=""):
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    ax.set_title(title, color=t["ink"], fontsize=12, fontweight="600", loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=8.5, va="bottom")
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-")   # hairline, solid
    ax.set_axisbelow(True)


def _legend(ax, t):
    lg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])          # text wears ink, the swatch carries identity
    return lg


def fig_range_gap(mode: str):
    """HEADLINE — radar's benefit by range bucket. The finding that survived."""
    t = THEME[mode]
    d = json.load(open("logs/radar_ablation.json"))
    buckets = ["near", "mid", "far"]
    xlabels = ["near\n0–20 m", "mid\n20–35 m", "far\n35–51 m"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, cond, cname in zip(axes, ("unseen_day", "unseen_night"), ("Daytime", "Night")):
        cam = [d[f"camera/{cond}"]["buckets"][b]["mAP"] for b in buckets]
        rad = [d[f"camera_radar/{cond}"]["buckets"][b]["mAP"] for b in buckets]
        x = np.arange(len(buckets))
        ax.bar(x - BAR_W / 2 - GAP, cam, BAR_W, label="camera only", color=t["s1"], zorder=3)
        ax.bar(x + BAR_W / 2 + GAP, rad, BAR_W, label="camera + radar", color=t["s2"], zorder=3)
        # Label only the far bucket — the story — and let the axis carry the rest.
        ax.text(2 - BAR_W / 2 - GAP, cam[2] + 0.006, f"{cam[2]:.3f}", ha="center",
                color=t["ink2"], fontsize=8.5)
        ax.text(2 + BAR_W / 2 + GAP, rad[2] + 0.006, f"{rad[2]:.3f}", ha="center",
                color=t["ink"], fontsize=8.5, fontweight="600")
        _style(ax, t, cname, "BEV mAP" if cond == "unseen_day" else "")
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
    axes[0].set_ylim(0, 0.30)
    # Legend at figure level, in the header band — inside the axes it collided
    # with the daytime bars, and nudging bars down to make room would have wasted
    # plot area to decoration.
    handles, labels_ = axes[0].get_legend_handles_labels()
    lg = fig.legend(handles, labels_, frameon=False, fontsize=9, ncol=2,
                    loc="upper right", bbox_to_anchor=(0.995, 1.045))
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])
    fig.suptitle("Radar closes the range gap, not the night gap",
                 color=t["ink"], fontsize=13, fontweight="700", x=0.008, ha="left", y=1.055)
    fig.text(0.008, 0.975, "Camera depth has effectively failed by 35 m even in daylight "
             "(mAP 0.012); radar holds 0.102.", color=t["ink2"], fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"range_gap_{mode}")


def fig_day_night(mode: str):
    """Phase 9 — the collapse, with the control that rules out dataset provenance."""
    t = THEME[mode]
    d = json.load(open("logs/day_night_audit.json"))["cells"]
    order = [("seen_day", "seen\nday", "day"), ("unseen_day", "unseen\nday", "day"),
             ("unseen_miniday", "unseen day\n(mini — control)", "day"),
             ("unseen_night", "unseen\nNIGHT", "night")]
    vals = [d[k]["mAP"] for k, _, _ in order]
    labels = [l for _, l, _ in order]
    # Colour encodes the variable under study — daylight vs night — not the bar's
    # rank or position. Three hues for one measure would imply four categories
    # that do not exist; two hues for a real binary is the honest encoding.
    colors = [t["s1"] if c == "day" else t["s2"] for _, _, c in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.42, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", color=t["ink"], fontsize=9,
                fontweight="600" if v == vals[-1] else "normal")
    _style(ax, t, "", "mAP")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(vals) * 1.20)

    import matplotlib.patches as mpatches
    lg = fig.legend(handles=[mpatches.Patch(color=t["s1"], label="daylight"),
                             mpatches.Patch(color=t["s2"], label="night")],
                    frameon=False, fontsize=9, ncol=2, loc="upper right",
                    bbox_to_anchor=(0.995, 1.045))
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])
    # Title and subtitle at figure level: at this panel height an axes-level
    # subtitle collides with the title.
    fig.suptitle("Detection collapses at night", color=t["ink"], fontsize=13,
                 fontweight="700", x=0.008, ha="left", y=1.055)
    fig.text(0.008, 0.975, "Same model, four conditions. The mini-day control scores like "
             "trainval-day, so the drop tracks lighting — not data source.",
             color=t["ink2"], fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"day_night_{mode}")


def fig_calibration(mode: str):
    """Phase 11 — reliability diagram: the introspection head vs the raw score."""
    t = THEME[mode]
    cell = json.load(open("logs/curve_data.json"))["unseen_night"]
    fig, ax = plt.subplots(figsize=(5.8, 4.9))
    ax.plot([0, 1], [0, 1], color=t["grid"], linewidth=1.5, zorder=2)
    diag_note = ax.text(0.0, 0.0, "perfect calibration", color=t["ink2"], fontsize=8,
                        rotation=45, rotation_mode="anchor")
    for key, name, col in (("head", "introspection head", t["s1"]),
                           ("raw", "raw detector score", t["s2"])):
        st = cell[key].get("bins_equal_mass", cell[key]["bins"])
        pts = [(c, a_) for c, a_, n in zip(st["confidence"], st["accuracy"], st["count"]) if n > 0]
        if not pts:
            continue
        conf, acc = zip(*pts)
        # 2px surface ring on the markers so they stay legible where the two
        # series cross each other or the diagonal.
        ax.plot(conf, acc, color=col, linewidth=2, marker="o", markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=2, label=name, zorder=4)
    _style(ax, t, "", "observed accuracy")
    ax.set_xlabel("predicted confidence", color=t["ink2"], fontsize=9)
    # Zoom to the data. At night essentially every detection scores below ~0.21,
    # so a full 0-1 reliability plot spends 79% of its area empty and the actual
    # deviation from the diagonal becomes invisible. The diagonal still reads as
    # the reference in a zoomed view.
    hi = max(max(cell[k]["bins_equal_mass"]["confidence"] or [0]) for k in ("head", "raw"))
    hi = max(hi, max(max(cell[k]["bins_equal_mass"]["accuracy"] or [0]) for k in ("head", "raw")))
    lim = min(1.0, hi * 1.25)
    ax.set_xlim(-lim * 0.03, lim); ax.set_ylim(-lim * 0.03, lim)
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    # lower-right is the empty quadrant on a reliability plot
    lg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])
    fig.suptitle("The trust layer is calibrated at night; the raw score is not",
                 color=t["ink"], fontsize=12, fontweight="700", x=0.008, ha="left", y=1.055)
    fig.text(0.008, 0.975, f"ECE {cell['head']['ece']:.3f} vs {cell['raw']['ece']:.3f}. "
             "Equal-mass bins. Below the diagonal = over-confident.",
             color=t["ink2"], fontsize=9, ha="left")
    diag_note.set_position((lim * 0.55, lim * 0.53))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"calibration_{mode}")


def fig_risk_coverage(mode: str):
    """Phase 11 — what abstention actually buys, the practitioner's question."""
    t = THEME[mode]
    cell = json.load(open("logs/curve_data.json"))["unseen_night"]
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    for key, name, col in (("head", "introspection head", t["s1"]),
                           ("raw", "raw detector score", t["s2"])):
        ax.plot(cell[key]["coverage"], cell[key]["risk"], color=col,
                linewidth=2, label=name, zorder=3)
    _style(ax, t, "", "error rate among kept detections")
    ax.set_xlabel("coverage (fraction of detections kept)", color=t["ink2"], fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    lg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])
    # Honest title: the two curves nearly coincide. Calling this "what abstention
    # buys" would imply a win the data does not show — at night the detector is
    # 97% false positives, so no ranking of those detections rescues much.
    fig.suptitle("Abstention barely helps when 97% of night detections are wrong",
                 color=t["ink"], fontsize=12, fontweight="700", x=0.008, ha="left", y=1.055)
    fig.text(0.008, 0.975,
             f"AURC {cell['head']['aurc']:.3f} vs {cell['raw']['aurc']:.3f} (lower is better) — "
             f"a real but small edge. Base error rate {1 - cell['positive_rate']:.3f}.",
             color=t["ink2"], fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"risk_coverage_{mode}")


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def main():
    for mode in ("light", "dark"):
        print(f"[{mode}]")
        for fn in (fig_range_gap, fig_day_night, fig_calibration, fig_risk_coverage):
            try:
                fn(mode)
            except (FileNotFoundError, KeyError) as e:
                print(f"  skip {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
