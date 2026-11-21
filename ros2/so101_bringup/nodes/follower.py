#!/usr/bin/env python3
"""Run the existing Xbox motion state machine from ROS timers and Joy messages."""

import math
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from common import AXES, BUTTONS, JOY_QOS, ArmTopics, configure_project, init_ros
from policy_input import PolicyInput


class JoyInput:
    name = "ROS Xbox input"

    def __init__(self, args, xbox, controller, timeout):
        self.xbox = xbox
        self.neutral = controller.neutral
        self.mapper = xbox.XboxActionMapper(args.deadzone, args.trigger_threshold)
        self.timeout = timeout
        self.state = xbox.XboxState()
        self.received_at = None
        self.armed = False

    def receive(self, message):
        if len(message.axes) != len(AXES) or len(message.buttons) != len(BUTTONS):
            return
        if any(not math.isfinite(value) or abs(value) > 1.0 for value in message.axes):
            return
        if any(value not in (0, 1) for value in message.buttons):
            return
        values = dict(zip(AXES, message.axes, strict=True))
        values.update(zip(BUTTONS, map(bool, message.buttons), strict=True))
        if not (0.0 <= values["left_trigger"] <= 1.0 and 0.0 <= values["right_trigger"] <= 1.0):
            return
        now = time.monotonic()
        if self.received_at is not None and now - self.received_at > self.timeout:
            self.armed = False
        self.received_at = now
        state = self.xbox.XboxState(timestamp=now, **values)
        if not self.armed:
            self.armed = self.neutral(state) and not state.lb and not state.menu
        self.state = state if self.armed else self.xbox.XboxState()

    @property
    def ready(self):
        return (self.armed and self.received_at is not None
                and time.monotonic() - self.received_at <= self.timeout)

    def connect(self):
        if not self.ready:
            raise self.xbox.XboxInputError("Waiting for a fresh neutral ROS controller sample.")

    def update(self):
        if not self.ready:
            self.armed = False
            self.state = self.xbox.XboxState()
        return self.state

    def get_action(self, *, update=False):
        return self.mapper.map_action(self.update() if update else self.state)

    def get_teleop_events(self):
        return self.mapper.map_events(self.state)

    def disconnect(self):
        self.armed = False


class Follower(Node):
    def __init__(self):
        super().__init__("follower")
        root = configure_project(self.declare_parameter("project_dir", os.getcwd()).value)
        import xbox_official_control as controller
        import xbox_teleoperator as xbox

        port = self.declare_parameter("port", "/dev/ttyACM0").value
        robot_id = self.declare_parameter("robot_id", "my_follower").value
        hz = self.declare_parameter("hz", 20.0).value
        execute = self.declare_parameter("enable_motion", False).value
        timeout = self.declare_parameter("joy_timeout", 0.25).value
        policy_mode = self.declare_parameter("policy_mode", "off").value
        policy_timeout = self.declare_parameter("policy_timeout", 0.75).value
        if policy_mode not in ("off", "preview", "live"):
            raise ValueError("policy_mode must be off, preview or live")
        if (policy_mode == "preview" and execute) or (policy_mode == "live" and not execute):
            raise ValueError("Preview requires motion disabled; live requires enable_motion")
        if not 0.1 <= policy_timeout <= 2.0:
            raise ValueError("policy_timeout must be within [0.1, 2.0] seconds")
        if not 0.1 <= timeout <= 1.0:
            raise ValueError("joy_timeout must be within [0.1, 1.0] seconds")
        args = controller.build_parser().parse_args([
            "control", "--port", port, "--robot-id", robot_id,
            "--hz", str(hz), "--calibration", str(root / "config" / f"{robot_id}.json"),
            *(["--execute"] if execute else []),
        ])
        controller.validate(args)
        self.policy = None if policy_mode == "off" else PolicyInput(
            self, controller.cart_module(), args, policy_mode, policy_timeout
        )
        self.topics = ArmTopics(self)
        self.joy = JoyInput(args, xbox, controller, timeout)
        self.create_subscription(Joy, "joy", self.joy.receive, JOY_QOS)
        self.controller = controller
        self.args = args
        self.steps = None
        self.finished = False
        self.create_timer(1.0 / hz, self.step)
        if policy_mode == "preview":
            self.get_logger().info("Policy preview ready; serial connection disabled.")
        elif execute:
            if policy_mode == "live":
                self.get_logger().info("Hold RB with the sticks and triggers released to apply fresh policy targets.")
            self.get_logger().info("Release the Xbox sticks, triggers and RB to connect the arm.")
        else:
            self.get_logger().info("Motion disabled. Waiting for Xbox input only.")

    def step(self):
        if not self.args.execute or self.finished:
            return
        if self.steps is None:
            if not self.joy.ready or not self.controller.neutral(self.joy.state):
                return
            self.steps = self.controller.control_steps(
                self.args, teleop_factory=lambda **_: self.joy,
                observation_callback=self.topics.position,
                action_callback=lambda values: self.topics.position(values, command=True),
                policy_target=self.policy.target if self.policy is not None else None,
            )
        try:
            next(self.steps)
        except StopIteration:
            self.finished = True

    def close(self):
        if self.steps is not None:
            self.steps.close()


def main():
    init_ros()
    node = None
    try:
        node = Follower()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
