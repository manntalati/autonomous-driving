"""
Phase 7 / P7-4 — parameter count + inference-speed benchmark.

Reports, per model in the unified pipeline, the parameter count and the
mean per-frame forward latency / FPS. Run after the demo config's checkpoints
exist.

    python -m demo.benchmark configs/demo.yaml
"""
from __future__ import annotations
import time
import yaml
import torch

from demo.pipeline import PerceptionPipeline


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def benchmark(cfg_path: str, num_iters: int = 20, warmup: int = 3) -> None:
    """
    Report per-model parameter counts and end-to-end pipeline latency / FPS.
    Args: cfg_path — configs/demo.yaml; num_iters — timed iterations; warmup — untimed.
    """
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    pipeline = PerceptionPipeline(cfg, device)

    models = {
        "temporal detector": pipeline.detector,
        "hybrid segmenter": pipeline.segmenter,
        "BEV detector": pipeline.bev,
    }
    print(f"\n{'model':24}{'parameters':>14}")
    for name, module in models.items():
        print(f"{name:24}{_count_params(module):>14,}")
    print(f"{'TOTAL':24}{sum(_count_params(m) for m in models.values()):>14,}")

    # end-to-end per-frame latency on a random 3-frame window
    seq_len = cfg.get("seq_len", 3)
    frames = torch.randn(seq_len, 3, 448, 800)
    intrinsic, cam_to_ego = torch.eye(3), torch.eye(4)
    for _ in range(warmup):
        pipeline.process_frame(frames, intrinsic, cam_to_ego)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        pipeline.process_frame(frames, intrinsic, cam_to_ego)
    per_frame = (time.perf_counter() - t0) / num_iters
    print(f"\nend-to-end: {per_frame * 1000:.1f} ms/frame  "
          f"({1.0 / per_frame:.2f} FPS) on {device}")


if __name__ == "__main__":
    import sys
    benchmark(sys.argv[1] if len(sys.argv) > 1 else "configs/demo.yaml")
