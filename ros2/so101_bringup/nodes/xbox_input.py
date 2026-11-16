#!/usr/bin/env python3
"""Publish the original Xbox controller state as a ROS Joy stream."""

import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from common import AXES, BUTTONS, JOY_QOS, configure_project, init_ros


class XboxInput(Node):
    def __init__(self):
        super().__init__("xbox_input")
        configure_project(self.declare_parameter("project_dir", os.getcwd()).value)
        from xbox_teleoperator import PygameXboxBackend

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        index = self.declare_parameter("controller_index", 0).value
        allow_other = self.declare_parameter("allow_non_xbox", False).value
        rate = self.declare_parameter("hz", 20.0).value
        if not 1.0 <= rate <= 30.0:
            raise ValueError("hz must be within [1, 30]")
        self.backend = PygameXboxBackend(index, not allow_other)
        self.publisher = self.create_publisher(Joy, "joy", JOY_QOS)
        self.create_timer(1.0 / rate, self.publish_input)
        self.connected = False

    def publish_input(self):
        from xbox_teleoperator import XboxInputError

        if not self.connected:
            try:
                self.backend.connect()
            except XboxInputError as error:
                self.get_logger().warning(str(error), throttle_duration_sec=5.0)
                return
            self.connected = True
            self.get_logger().info(f"Controller connected: {self.backend.name}")
        try:
            state = self.backend.read()
        except XboxInputError as error:
            self.get_logger().warning(f"Controller disconnected: {error}")
            self.backend.disconnect()
            self.connected = False
            return
        message = Joy()
        message.header.stamp = self.get_clock().now().to_msg()
        message.axes = [float(getattr(state, field)) for field in AXES]
        message.buttons = [int(getattr(state, field)) for field in BUTTONS]
        self.publisher.publish(message)


def main():
    init_ros()
    node = None
    try:
        node = XboxInput()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.backend.disconnect()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
