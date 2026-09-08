#!/usr/bin/env bash
# Export YOLO26 weights to a TensorRT engine on the TARGET machine.
# Usage: bash scripts/export_engine.sh [weights.pt] [out.engine] [imgsz]
set -euo pipefail

WEIGHTS=${1:-yolo26n.pt}
ENGINE=${2:-yolo26n.engine}
IMGSZ=${3:-640}

if [[ ! -f "$WEIGHTS" ]]; then
  echo "weights not found: $WEIGHTS" >&2
  exit 1
fi

if ! command -v yolo >/dev/null 2>&1; then
  echo "install ultralytics first: pip install ultralytics" >&2
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not in PATH. On Jetson it is often:" >&2
  echo "  /usr/src/tensorrt/bin/trtexec" >&2
  exit 1
fi

STEM=${WEIGHTS%.pt}
ONNX="${STEM}.onnx"

echo "==> ONNX: $WEIGHTS -> $ONNX (imgsz=$IMGSZ)"
yolo export model="$WEIGHTS" format=onnx imgsz="$IMGSZ"

if [[ ! -f "$ONNX" ]]; then
  # Ultralytics sometimes writes next to the model with a different stem
  found=$(ls -1 ./*.onnx 2>/dev/null | head -n1 || true)
  if [[ -n "$found" ]]; then
    ONNX=$found
  else
    echo "onnx not produced" >&2
    exit 1
  fi
fi

echo "==> TensorRT FP16: $ONNX -> $ENGINE"
trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --fp16
echo "done: $ENGINE"
echo "expect end-to-end output shape [1, max_det, 6]"
