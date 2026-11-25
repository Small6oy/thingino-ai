#!/usr/bin/env bash
# Fine-tune the openly licensed detectors for three classes: person, pet, vehicle (no package).
# For a GPU machine. Every model it makes starts from Apache-2.0 weights and open data only,
# so the results can be published.
#
# Usage: scripts/train-open-3class.sh [yoloxn|mnv4n|both] [extra build_model.py options...]
#   yoloxn  YOLOX-Nano (Megvii, Apache-2.0), the default and first choice
#   mnv4n   MobileNetV4-conv-small-0.5 (timm, Apache-2.0) + YOLOX-Nano neck/heads
#   both    yoloxn, then mnv4n
# e.g.  scripts/train-open-3class.sh
#       scripts/train-open-3class.sh both --coco-root /data/coco2017 --device cuda
#       SKIP_QUICK=1 scripts/train-open-3class.sh yoloxn --epochs 40
#       INPUT=384x224 scripts/train-open-3class.sh yoloxn   # 16:9 input, ~40 % faster on the T31
#
# For each model it runs build_model.py --quick first (~10 min: catches setup problems before the
# long run), then the full build: a float fine-tune (20 epochs) re-learns around the T31 changes
# (ReLU, 3x3 stem, no SPP), then quantisation-aware training (30 epochs), then both exports.
# Untrained, these models score AP50 ~0 on the camera: the fine-tune is what makes them work.
# Same command again = resume. Results: train/out/<run>/ (model_t31.bin, model_t41.onnx,
# model_t41.bin when Docker on x86-64 is available, report.json, log.txt, checkpoint.pt).
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
which=${1:-yoloxn}
[ $# -gt 0 ] && shift
case "$which" in
  yoloxn) archs=(yoloxn) ;;
  mnv4n) archs=(mnv4n) ;;
  both) archs=(yoloxn mnv4n) ;;
  *) echo "usage: $0 [yoloxn|mnv4n|both] [build_model.py options...]" >&2; exit 2 ;;
esac

classes=person,pet,vehicle
size=()
suffix=
if [ -n "${INPUT:-}" ]; then
  size=(--input "$INPUT")
  suffix=-$INPUT
fi
py=${PYTHON:-python3}
cd "$here/train"

for arch in "${archs[@]}"; do
  name=open3-$arch$suffix
  if [ -z "${SKIP_QUICK:-}" ] && [ ! -f "out/$name/report.json" ]; then
    echo "=== $arch: quick smoke test ($classes)"
    "$py" build_model.py --quick --arch "$arch" --classes "$classes" ${size[@]+"${size[@]}"} --name "$name-quick" "$@"
  fi
  echo "=== $arch: full build ($classes)"
  "$py" build_model.py --arch "$arch" --classes "$classes" ${size[@]+"${size[@]}"} --name "$name" "$@"
  echo "=== $arch: done. Send us train/out/$name/ (model_t31.bin, model_t41.onnx, model_t41.bin, report.json, log.txt, checkpoint.pt)"
done
