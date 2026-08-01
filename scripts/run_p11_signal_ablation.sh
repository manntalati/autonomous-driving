#!/usr/bin/env bash
# Phase 11 signal ablation — does epistemic uncertainty beat the raw detector score?
#
# THE COMPARISON MUST SIT ON ONE MODEL
# ------------------------------------
# The earlier P11 result used detector_best.pt (no dropout, mAP 0.285) and had no
# epistemic features at all. Comparing it against an MC-dropout run on
# detector_dropout_best.pt (mAP 0.274) would span two detectors of different
# capability, and a difference in mAP would masquerade as a difference in signal
# value. So BOTH arms below run on the dropout checkpoint; the only variable is
# whether epistemic features are present.
#
#   arm A: --mc-samples 0   -> geometry + class + score        (baseline)
#   arm B: --mc-samples 20  -> the same, plus score_var/box_var (epistemic)
#
# Runs on CPU by default (P11_DEVICE=cpu) so it can share the machine with the
# Phase 10 BEV training without both jobs contending for the single MPS device.
# Override with P11_DEVICE=mps when nothing else is running.
#
# Usage: bash scripts/run_p11_signal_ablation.sh

set -u
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

CKPT=checkpoints/detector_dropout_best.pt
CFG=configs/detector_dropout.yaml

if [ ! -f "$CKPT" ]; then
  echo "missing $CKPT — run the dropout fine-tune first"; exit 1
fi

echo "=== [P11 arm A: no epistemic] start $(date '+%F %T') ==="
.venv/bin/python -m models.uncertainty.train_introspection "$CFG" \
  --ckpt "$CKPT" --mc-samples 0 --device "${P11_DEVICE:-cpu}" \
  --out checkpoints/introspection_nomc.pt \
  --report logs/introspection_nomc.json > logs/p11_arm_a.log 2>&1
echo "=== [arm A] exit $? at $(date '+%F %T') ==="
tail -8 logs/p11_arm_a.log

echo "=== [P11 arm B: MC-dropout, 20 samples] start $(date '+%F %T') ==="
echo "    (20x the forward cost — this is the long one)"
.venv/bin/python -m models.uncertainty.train_introspection "$CFG" \
  --ckpt "$CKPT" --mc-samples 20 --device "${P11_DEVICE:-cpu}" \
  --out checkpoints/introspection_mc.pt \
  --report logs/introspection_mc.json > logs/p11_arm_b.log 2>&1
echo "=== [arm B] exit $? at $(date '+%F %T') ==="
tail -8 logs/p11_arm_b.log

echo "=== [P11] both arms done $(date '+%F %T') ==="
