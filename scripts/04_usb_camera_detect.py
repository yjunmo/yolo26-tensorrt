#!/usr/bin/env python3
"""USB 摄像头实时检测。无显示器时加 --no-show --save out.mp4。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USB camera YOLO detect")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default="0", help="摄像头编号或视频路径")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default=0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save", default="", help="可选，保存带框视频")
    return p.parse_args()


def open_source(src: str) -> cv2.VideoCapture:
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先: python3 -m pip install ultralytics opencv-python-headless", file=sys.stderr)
        return 1

    model = YOLO(args.weights)
    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"打不开视频源: {args.source}，试试 --source 1 或检查 ls /dev/video*", file=sys.stderr)
        return 1

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (frame_w, frame_h))

    print("按 q 退出（有窗口时）")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        vis = result.plot()
        if writer is not None:
            if vis.shape[1] != frame_w or vis.shape[0] != frame_h:
                vis = cv2.resize(vis, (frame_w, frame_h))
            writer.write(vis)
        if not args.no_show:
            cv2.imshow("yolo", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"已保存 {Path(args.save).resolve()}")
    if not args.no_show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
