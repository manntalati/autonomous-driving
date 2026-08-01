#!/usr/bin/env bash
# Phase 10 ablation training — runs BOTH arms back to back on one device.
#
# WHY RETRAIN THE CAMERA ARM
# --------------------------
# checkpoints/bev_surround_best.pt already exists and still loads strict=True, so
# it is tempting to use it as the camera baseline for free. We do not, because the
# Phase 10 claim is a mAP difference of order 0.02, and the existing checkpoint was
# trained two months earlier against a dataset and loss that have since changed.
# A confound that size could manufacture the entire result. Both arms are therefore
# trained here with the same code, the same seed (seed: 0) and the same protocol, differing
# only in the radar block of the config.
#
# Sequential, not parallel: one MPS device, and two concurrent jobs would contend
# for it and distort the per-epoch timings we report.
#
# FIXED-EPOCH, FINAL-CHECKPOINT COMPARISON
# ----------------------------------------
# Both arms train for exactly 12 epochs with early stopping disabled, and the
# ablation evaluates the FINAL checkpoints (*_last.pt), not the best-by-mIoU ones.
# The first attempt early-stopped on BEV mIoU and the camera arm selected epoch 2
# out of 12, purely because that noisy metric happened to peak there
# (.124 .186 .120 .160 .142 .124 .136 .184 .139 .145 .154 .131 — no trend).
# Selecting each arm on a coin flip would have let a difference in training
# maturity masquerade as the radar effect.
#
# Usage:  bash scripts/run_p10_ablation.sh [pid-to-wait-for]

set -u
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[chain] waiting for PID $WAIT_PID to finish before starting..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[chain] PID $WAIT_PID done."
fi

run () {
  local name="$1" cfg="$2" log="$3"
  echo "=== [$name] start $(date '+%F %T') ==="
  .venv/bin/python -m models.bev.train_bev "$cfg" > "$log" 2>&1
  local rc=$?
  echo "=== [$name] exit $rc at $(date '+%F %T') ==="
  # Surface the outcome without dumping the whole log.
  grep -E "^Epoch|Early stopping|Traceback|Error" "$log" | tail -3
  return $rc
}

# Camera-only arm first: it is the baseline, and if it fails there is no point
# spending hours on the radar arm.
run "camera-only" configs/bev_surround_p10.yaml logs/bev_surround_p10_run.log \
  || { echo "[chain] camera arm failed — stopping before the radar arm"; exit 1; }

run "camera+radar" configs/bev_radar.yaml logs/bev_radar_run.log \
  || { echo "[chain] radar arm failed"; exit 1; }

echo "[chain] both arms done — running the 2x2 ablation"
.venv/bin/python -m evaluation.radar_ablation \
  --camera-config configs/bev_surround_p10.yaml \
  --camera-ckpt checkpoints/bev_surround_p10_last.pt \
  --radar-config configs/bev_radar.yaml \
  --radar-ckpt checkpoints/bev_radar_last.pt \
  > logs/radar_ablation_run.log 2>&1
echo "[chain] ablation exit $? — see logs/radar_ablation_run.log"
tail -20 logs/radar_ablation_run.log
