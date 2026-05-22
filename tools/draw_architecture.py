"""
Generate one architecture diagram per phase — each in a style suited to that
network (slab stack, feature pyramid, U-shape, transformer flowchart, lift-splat
schematic, temporal-fusion graph, pipeline fan-out). Every element is annotated
with its tensor dimensions (channels × height × width).

    python -m tools.draw_architecture     # → docs/diagrams/*.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle

OUT_DIR = Path("docs/diagrams")

# palette
C_INPUT = "#e3e8ee"
C_CONV = "#bcd4ec"
C_CONV_D = "#8fb4dc"
C_POOL = "#f0c4c4"
C_ATTN = "#d9c8ec"
C_HEAD = "#c9e3c9"
C_GEOM = "#f3dab0"
EDGE = "#2b2b2b"


# ── primitives ───────────────────────────────────────────────────────────────
def _box(ax, cx, cy, w, h, text, fc=C_CONV, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012", facecolor=fc,
                                edgecolor=EDGE, lw=1.4, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            weight=weight, zorder=4)


def _arrow(ax, p0, p1, color=EDGE, lw=1.6, rad=0.0):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2, mutation_scale=14))


def _label(ax, x, y, text, fs=8, color="#444", ha="center", va="top", weight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, weight=weight, zorder=4)


def _iso_slab(ax, cx, h, depth, fc, face_w=0.6):
    """3D isometric slab centred at cx, on baseline y=0."""
    ox, oy = 0.40 * depth, 0.30 * depth
    x0, x1 = cx - face_w / 2, cx + face_w / 2
    ax.add_patch(Polygon([(x1, 0), (x1, h), (x1 + ox, h + oy), (x1 + ox, oy)],
                          closed=True, facecolor="#7f8c99", edgecolor=EDGE, lw=1.2, zorder=2))
    ax.add_patch(Polygon([(x0, h), (x1, h), (x1 + ox, h + oy), (x0 + ox, h + oy)],
                          closed=True, facecolor="#b9c4cf", edgecolor=EDGE, lw=1.2, zorder=2))
    ax.add_patch(Polygon([(x0, 0), (x1, 0), (x1, h), (x0, h)],
                          closed=True, facecolor=fc, edgecolor=EDGE, lw=1.5, zorder=3))


def _new(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    ax.axis("off")
    return fig, ax


def _save(fig, ax, name, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ── Phase 1 — ResNet backbone: 3D slab stack (VGG-style) ─────────────────────
def draw_backbone():
    fig, ax = _new(13, 6.2, "Phase 1 — ResNet-18 Backbone  (3D feature-volume stack)")
    stages = [
        ("Input",            "3 × 448 × 800",   3.30, 0.40, C_INPUT),
        ("Stem\n7×7 s2 + pool", "64 × 112 × 200", 2.40, 0.70, C_POOL),
        ("Stage 1\n2× residual", "64 × 112 × 200", 2.40, 0.70, C_CONV),
        ("Stage 2 · C3\n2× residual, s2", "128 × 56 × 100", 1.80, 0.95, C_CONV),
        ("Stage 3 · C4\n2× residual, s2", "256 × 28 × 50",  1.25, 1.20, C_CONV_D),
        ("Stage 4 · C5\n2× residual, s2", "512 × 14 × 25",  0.85, 1.50, C_CONV_D),
    ]
    spacing = 2.45
    centers = [1.0 + i * spacing for i in range(len(stages))]
    top = 0
    for cx, (label, dims, h, d, fc) in zip(centers, stages):
        _iso_slab(ax, cx, h, d, fc)
        top = max(top, h + 0.30 * d)
        _label(ax, cx, -0.45, label, fs=10, weight="bold", color="black")
        _label(ax, cx, -1.45, dims, fs=9)
    for i in range(len(stages) - 1):
        _arrow(ax, (centers[i] + 0.55, top * 0.42), (centers[i + 1] - 0.45, top * 0.42),
               color="#5b6b7a")
    _save(fig, ax, "phase1_backbone", (-0.5, centers[-1] + 1.7), (-2.4, top + 0.7))


# ── Phase 2 — FPN detector: feature-pyramid flowchart ────────────────────────
def draw_detector():
    fig, ax = _new(11, 7.6, "Phase 2 — FPN Single-Stage Detector  (feature pyramid)")
    # backbone column (left) — C3 bottom (large) → C5 top (small)
    bx = 1.6
    cs = [("C5", "512 × 14 × 25", 6.2, 1.5, 0.7),
          ("C4", "256 × 28 × 50", 4.2, 2.0, 0.9),
          ("C3", "128 × 56 × 100", 2.0, 2.6, 1.1)]
    for name, dims, cy, w, h in cs:
        _box(ax, bx, cy, w, h, f"{name}\n{dims}", fc=C_CONV, fs=8.5)
    _label(ax, bx, -0.4, "ResNet-18 backbone", fs=9, weight="bold")
    # FPN column (middle) — P-levels, 256-ch
    px = 5.7
    ps = [("P5", "256 × 14 × 25", 6.2), ("P4", "256 × 28 × 50", 4.2), ("P3", "256 × 56 × 100", 2.0)]
    for name, dims, cy in ps:
        _box(ax, px, cy, 2.0, 1.1, f"{name}\n{dims}", fc=C_CONV_D, fs=8)
    _label(ax, px, -0.4, "FPN  (256-ch pyramid)", fs=9, weight="bold")
    # lateral (C→P) + top-down (P5→P4→P3)
    for (_, _, cy, w, _), (_, _, py) in zip(cs, ps):
        _arrow(ax, (bx + w / 2, cy), (px - 1.0, py), color="#5b6b7a")
    _arrow(ax, (px, 6.2 - 0.55), (px, 4.2 + 0.55), color="#c0392b", lw=2.0)
    _arrow(ax, (px, 4.2 - 0.55), (px, 2.0 + 0.55), color="#c0392b", lw=2.0)
    _label(ax, px + 1.2, 5.2, "top-down\n+ upsample", fs=7.5, color="#c0392b", ha="left", va="center")
    # shared head + outputs (right)
    hx = 8.9
    _box(ax, hx, 4.2, 2.0, 2.4, "Detection head\nshared 4-conv tower", fc=C_HEAD, fs=8.5)
    for _, _, py in ps:
        _arrow(ax, (px + 1.0, py), (hx - 1.0, 4.2), color="#5b6b7a")
    _box(ax, hx, 1.4, 2.0, 1.0, "cls logits + box deltas\n→ NMS → mAP", fc=C_INPUT, fs=8)
    _arrow(ax, (hx, 4.2 - 1.2), (hx, 1.4 + 0.5))
    _save(fig, ax, "phase2_detector", (-0.2, 10.2), (-1.2, 7.6))


# ── Phase 3 — U-Net: U-shape encoder/decoder ─────────────────────────────────
def draw_unet():
    fig, ax = _new(10.5, 7.8, "Phase 3 — U-Net Semantic Segmentation  (encoder–decoder)")
    xe, xd = 2.4, 8.2                       # encoder / decoder columns
    L0, L1, L2, L3 = 6.2, 4.7, 3.2, 1.7     # resolution levels (top → bottom)
    bw, bh = 3.0, 1.0
    enc = [("Encoder · Image", "3 × 448 × 800", L0),
           ("Encoder · C3",    "128 × 56 × 100", L1),
           ("Encoder · C4",    "256 × 28 × 50",  L2),
           ("Encoder · C5",    "512 × 14 × 25",  L3)]
    dec = [("Up-block 3", "256 × 28 × 50",   L2),
           ("Up-block 2", "128 × 56 × 100",  L1),
           ("Up-block 1", "64 × 112 × 200",  L0)]
    for name, dims, cy in enc:
        _box(ax, xe, cy, bw, bh, f"{name}\n{dims}", fc=C_CONV, fs=8)
    for name, dims, cy in dec:
        _box(ax, xd, cy, bw, bh, f"{name}\n{dims}", fc=C_CONV_D, fs=8)
    _box(ax, xd, 7.7, bw, bh, "Seg logits\n5 × 448 × 800", fc=C_HEAD, fs=8)
    # encoder downsampling (down the left column)
    for i in range(len(enc) - 1):
        _arrow(ax, (xe, enc[i][2] - bh / 2), (xe, enc[i + 1][2] + bh / 2))
    # bottleneck C5 → Up-block 3 (the one diagonal, across the bottom of the U)
    _arrow(ax, (xe + bw / 2, L3), (xd - bw / 2, L2))
    # decoder upsampling (up the right column)
    for y0, y1 in [(L2, L1), (L1, L0), (L0, 7.7)]:
        _arrow(ax, (xd, y0 + bh / 2), (xd, y1 - bh / 2))
    # skip connections — horizontal, at matching resolution levels
    for ey, txt in [(L2, "skip · C4 concat"), (L1, "skip · C3 concat")]:
        _arrow(ax, (xe + bw / 2, ey), (xd - bw / 2, ey), color="#c0392b", lw=1.9)
        _label(ax, (xe + xd) / 2, ey + 0.18, txt, fs=8, color="#c0392b", va="bottom")
    _label(ax, xe, L3 - bh / 2 - 0.30, "encoder = ResNet backbone", fs=8.5, weight="bold")
    _label(ax, xd, L2 - bh / 2 - 0.30,
           "decoder = UpBlock (bilinear ↑ → concat skip → 2 ConvBlocks)", fs=8.5, weight="bold")
    _save(fig, ax, "phase3_unet", (-0.6, 11.6), (0.4, 9.0))


# ── Phase 4 — ViT encoder block: pre-norm transformer flowchart ──────────────
def draw_vit():
    fig, ax = _new(8.2, 9.4, "Phase 4 — Vision Transformer Encoder Block  (pre-norm)")
    cx = 2.6
    nodes = [
        ("Patch embed + pos\n1400 tokens × 384-d", 8.4, C_GEOM),
        ("LayerNorm", 7.1, C_INPUT),
        ("Multi-Head Self-Attention\n6 heads × 64-d", 5.9, C_ATTN),
        ("⊕  residual", 4.8, C_INPUT),
        ("LayerNorm", 3.7, C_INPUT),
        ("MLP   384 → 1536 → 384\nGELU", 2.5, C_HEAD),
        ("⊕  residual", 1.4, C_INPUT),
    ]
    for text, cy, fc in nodes:
        w = 1.0 if "⊕" in text else 3.6
        h = 0.7 if "⊕" in text else 0.92
        _box(ax, cx, cy, w, h, text, fc=fc, fs=8.5)
    for i in range(len(nodes) - 1):
        _arrow(ax, (cx, nodes[i][1] - (0.35 if "⊕" in nodes[i][0] else 0.46)),
                   (cx, nodes[i + 1][1] + (0.35 if "⊕" in nodes[i + 1][0] else 0.46)))
    # skip arrows around the two residual adds
    _arrow(ax, (cx + 1.8, 7.1), (cx + 0.5, 4.8), color="#c0392b", lw=1.7, rad=-0.55)
    _arrow(ax, (cx + 1.8, 3.7), (cx + 0.5, 1.4), color="#c0392b", lw=1.7, rad=-0.55)
    _label(ax, cx + 2.4, 6.0, "skip", fs=8, color="#c0392b", ha="left", va="center")
    _label(ax, cx + 2.4, 2.6, "skip", fs=8, color="#c0392b", ha="left", va="center")
    _label(ax, cx - 2.3, 4.9, "×6\nblocks", fs=9, color="#333", ha="right", va="center", weight="bold")
    _save(fig, ax, "phase4_vit", (-1.0, 7.0), (0.6, 9.2))


# ── Phase 5 — Lift-Splat-Shoot: geometric schematic ──────────────────────────
def draw_bev():
    fig, ax = _new(11.5, 6.6, "Phase 5 — Lift-Splat-Shoot BEV Transform  (geometry)")
    # camera + image feature map
    _box(ax, 1.3, 4.3, 1.8, 1.5, "C4 features\n256 × 28 × 50", fc=C_CONV, fs=8)
    _box(ax, 1.3, 2.3, 1.8, 0.9, "DepthNet\ndepth dist. (46 bins)", fc=C_INPUT, fs=7.5)
    _arrow(ax, (1.3, 3.55), (1.3, 2.78))
    # LIFT — frustum trapezoid
    fx = 3.2
    ax.add_patch(Polygon([(fx, 3.4), (fx, 5.2), (fx + 3.4, 6.4), (fx + 3.4, 2.2)],
                          closed=True, facecolor=C_GEOM, edgecolor=EDGE, lw=1.5, alpha=0.85, zorder=3))
    for k in range(1, 5):                       # depth-bin slices
        t = k / 5
        ax.plot([fx + 3.4 * t, fx + 3.4 * t],
                [3.4 + (2.2 - 3.4) * t * 0 + (3.4 - (3.4 + 1.0 * t)),  # noqa
                 5.2 + (6.4 - 5.2) * t], lw=0)   # (kept simple; slices drawn below)
    for k in range(1, 5):
        t = k / 5.0
        y0 = 3.4 - 1.2 * t
        y1 = 5.2 + 1.2 * t
        ax.plot([fx + 3.4 * t, fx + 3.4 * t], [y0, y1], color="#b07a1e", lw=1.0, zorder=4)
    _label(ax, fx + 1.7, 1.7, "LIFT — feature ⊗ depth\n→ (256 × 46 × 28 × 50) frustum",
           fs=8, weight="bold")
    _arrow(ax, (2.25, 4.3), (fx + 0.05, 4.3))
    # SPLAT — BEV grid
    gx, gy, gs = 7.7, 2.3, 3.0
    n = 8
    for i in range(n + 1):
        ax.plot([gx + gs * i / n] * 2, [gy, gy + gs], color="#888", lw=0.6, zorder=2)
        ax.plot([gx, gx + gs], [gy + gs * i / n] * 2, color="#888", lw=0.6, zorder=2)
    ax.add_patch(Rectangle((gx, gy), gs, gs, facecolor="none", edgecolor=EDGE, lw=1.6, zorder=3))
    ax.add_patch(Circle((gx + gs / 2, gy + gs / 2), 0.10, facecolor="black", zorder=4))
    _label(ax, gx + gs / 2, gy - 0.25, "SPLAT — scatter to BEV grid\n128 × 128 × 64",
           fs=8, weight="bold")
    _arrow(ax, (fx + 3.5, 4.3), (gx - 0.15, gy + gs / 2), color="#5b6b7a")
    # BEV head
    _box(ax, gx + gs + 1.6, gy + gs / 2, 1.9, 1.4,
         "BEV head\nheatmap (3)\n+ box reg (6)", fc=C_HEAD, fs=8)
    _arrow(ax, (gx + gs, gy + gs / 2), (gx + gs + 0.65, gy + gs / 2))
    _save(fig, ax, "phase5_bev", (-0.2, 14.2), (1.0, 7.0))


# ── Phase 6 — temporal cross-attention: fusion graph ─────────────────────────
def draw_temporal():
    fig, ax = _new(10.5, 6.8, "Phase 6 — Temporal Cross-Attention Detector  (3-frame fusion)")
    frames = [("Frame t-2", 5.4), ("Frame t-1", 3.4), ("Frame t  (current)", 1.4)]
    for name, cy in frames:
        _box(ax, 1.5, cy, 2.2, 0.9, f"{name}\n3 × 448 × 800", fc=C_INPUT, fs=7.8)
        _box(ax, 4.4, cy, 1.9, 0.9, "Backbone\nC5  512×14×25", fc=C_CONV, fs=7.8)
        _arrow(ax, (2.6, cy), (3.45, cy))
    # cross-attention block
    ax_cx, ax_cy = 7.2, 3.4
    _box(ax, ax_cx, ax_cy, 2.4, 2.6,
         "Cross-Attention\n\nQ ← frame t\nK,V ← {t-1, t-2}\n+ temporal embed", fc=C_ATTN, fs=8)
    _arrow(ax, (5.35, 5.4), (ax_cx - 1.2, ax_cy + 0.7), color="#5b6b7a")   # t-2 → K/V
    _arrow(ax, (5.35, 3.4), (ax_cx - 1.2, ax_cy), color="#5b6b7a")          # t-1 → K/V
    _arrow(ax, (5.35, 1.4), (ax_cx - 1.2, ax_cy - 0.9), color="#c0392b", lw=1.9)  # t → Q
    _label(ax, 6.0, 1.0, "red = query (current frame)", fs=7.5, color="#c0392b")
    # fused → FPN/head → output
    _box(ax, ax_cx + 3.0, ax_cy, 1.9, 1.3, "Fused C5\n→ FPN + head\n(Phase 2)", fc=C_HEAD, fs=8)
    _arrow(ax, (ax_cx + 1.2, ax_cy), (ax_cx + 3.0 - 0.95, ax_cy))
    _save(fig, ax, "phase6_temporal", (0.0, 12.4), (0.4, 6.9))


# ── Phase 7 — unified pipeline: shared trunk → parallel heads ────────────────
def draw_pipeline():
    fig, ax = _new(11.5, 6.4, "Phase 7 — Unified Perception Pipeline")
    _box(ax, 1.4, 3.2, 2.0, 1.3, "Frame window\n3 × 448 × 800", fc=C_INPUT, fs=8)
    _box(ax, 4.2, 3.2, 2.0, 1.3, "Shared\nResNet backbone\nC3 / C4 / C5", fc=C_CONV, fs=8)
    _arrow(ax, (2.45, 3.2), (3.15, 3.2))
    heads = [("Temporal detector", "→ 2D boxes (3 cls)", 5.4, C_HEAD),
             ("U-Net segmenter", "→ mask (5 cls)", 3.2, C_HEAD),
             ("LSS BEV detector", "→ top-down boxes", 1.0, C_HEAD)]
    for name, out, cy, fc in heads:
        _box(ax, 7.4, cy, 2.4, 1.2, f"{name}\n{out}", fc=fc, fs=8)
        _arrow(ax, (5.25, 3.2), (6.15, cy), color="#5b6b7a")
        _arrow(ax, (8.65, cy), (9.55, 3.2), color="#5b6b7a")
    _box(ax, 10.7, 3.2, 2.0, 1.5, "Unified output\ndetections +\nsegmentation +\nBEV", fc=C_GEOM, fs=8)
    _label(ax, 7.4, -0.2, "the three heads run in parallel on the shared backbone features",
           fs=8, color="#444")
    _save(fig, ax, "phase7_pipeline", (0.0, 12.2), (-0.6, 6.4))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_backbone()
    draw_detector()
    draw_unet()
    draw_vit()
    draw_bev()
    draw_temporal()
    draw_pipeline()


if __name__ == "__main__":
    main()
