#!/usr/bin/env python3
"""在当前设备上测 YOLO 推理 FPS（含预处理，更接近真实）。"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark YOLO FPS")
    p.add_argument("--weights", required=True, help=".pt / .engine / .onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先: python3 -m pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(args.weights)
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)
    dt = time.perf_counter() - t0
    fps = args.iters / dt if dt > 0 else 0.0
    print(f"weights={args.weights}")
    print(f"imgsz={args.imgsz}  iters={args.iters}  time={dt:.3f}s")
    print(f"FPS={fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
