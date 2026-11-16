#!/usr/bin/env python3
"""Publish the two UVC cameras in USB stream startup order."""

from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from common import init_ros


class UvcCameras(Node):
    def __init__(self):
        super().__init__("cameras")
        devices = {
            "wrist": self.declare_parameter("wrist_device", "/dev/video0").value,
            "top": self.declare_parameter("top_device", "/dev/video2").value,
        }
        if Path(devices["wrist"]).resolve() == Path(devices["top"]).resolve():
            raise ValueError("wrist_device and top_device must refer to different cameras")
        width = self.declare_parameter("width", 640).value
        height = self.declare_parameter("height", 360).value
        fps = self.declare_parameter("fps", 30.0).value
        fourcc = self.declare_parameter("fourcc", "YUYV").value
        if fps <= 0 or width <= 0 or height <= 0 or len(fourcc) != 4:
            raise ValueError("Invalid camera size, frame rate or pixel format")

        self.cameras = {}
        self.image_publishers = {}
        try:
            for view, device in devices.items():
                cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                self.cameras[view] = cap
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open {device}")
                for key, value in (
                    (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
                    (cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                    (cv2.CAP_PROP_FPS, fps),
                ):
                    cap.set(key, value)
                self.image_publishers[view] = self.create_publisher(
                    Image, f"{view}/image_raw", qos_profile_sensor_data
                )
                self.get_logger().info(
                    f"{view}: {device}, requested {width}x{height}, {fps:g} FPS, {fourcc}"
                )
            self.create_timer(1.0 / fps, self.publish_frames)
        except BaseException:
            self.close()
            raise

    def publish_frames(self):
        for view, cap in self.cameras.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                self.get_logger().warning(f"{view}: no video frame", throttle_duration_sec=5.0)
                continue
            message = Image()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = f"{view}_camera_optical_frame"
            message.height, message.width = frame.shape[:2]
            message.encoding = "bgr8"
            message.step = message.width * 3
            message.data = frame.tobytes()
            self.image_publishers[view].publish(message)

    def close(self):
        for cap in self.cameras.values():
            cap.release()


def main():
    init_ros()
    node = None
    try:
        node = UvcCameras()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
