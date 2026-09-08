#!/usr/bin/env python3
"""订阅 sensor_msgs/Image，发布带框图像。只处理队列里最新的一帧。"""
from __future__ import annotations

from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class YoloDetectNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detect")
        self.declare_parameter("weights", "yolov8n.pt")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("output_topic", "/yolo/image")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("device", "0")

        weights = str(self.get_parameter("weights").value)
        image_topic = str(self.get_parameter("image_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.device = str(self.get_parameter("device").value)

        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.bridge = CvBridge()
        self._latest: Optional[Image] = None
        self.pub = self.create_publisher(Image, output_topic, 10)
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_timer(0.05, self._spin_once)
        self.get_logger().info(f"weights={weights}  sub={image_topic}  pub={output_topic}")

    def _on_image(self, msg: Image) -> None:
        self._latest = msg

    def _spin_once(self) -> None:
        msg = self._latest
        self._latest = None
        if msg is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        vis = result.plot()
        out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
