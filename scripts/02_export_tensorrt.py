#!/usr/bin/env python3
"""在 Jetson 本机把 YOLO .pt 导出为 TensorRT engine。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Ultralytics YOLO to TensorRT")
    p.add_argument("--weights", required=True, help=".pt 权重路径")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16（推荐）")
    p.add_argument("--int8", action="store_true", help="INT8，需要 --data")
    p.add_argument("--data", default="", help="INT8 校准用的 data yaml")
    p.add_argument("--device", default=0)
    p.add_argument("--workspace", type=int, default=4, help="TensorRT workspace GB")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        print(f"找不到权重: {weights}", file=sys.stderr)
        return 1
    if args.int8 and not args.data:
        print("INT8 必须提供 --data xxx.yaml", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先: python3 -m pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(str(weights))
    kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        half=bool(args.half and not args.int8),
        int8=bool(args.int8),
        device=args.device,
        workspace=args.workspace,
        simplify=True,
    )
    if args.int8:
        kwargs["data"] = args.data

    print("开始导出（请在 Jetson 上运行，可能要几分钟）...")
    out = model.export(**kwargs)
    print(f"完成: {out}")
    print("接着测速: python3 scripts/03_bench_infer.py --weights <刚才的.engine>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
