"""Shared ROS I/O and access to the project's hardware code."""

import math
import os
from pathlib import Path
import sys

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

AXES = ("left_x", "left_y", "right_x", "right_y", "left_trigger", "right_trigger")
JOY_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
BUTTONS = ("a", "b", "x", "y", "lb", "rb", "view", "menu", "left_stick", "right_stick")


def configure_project(path):
    root = Path(path).expanduser().resolve()
    if not (root / "so101_cartesian_control.py").is_file():
        raise ValueError(f"Specify the SO-101 checkout with project_dir: {root}")
    for folder in (root, root / "xbox_controller"):
        sys.path.insert(0, str(folder))
    os.environ.setdefault("HF_HOME", str(root / ".hf-home"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(root / ".lerobot"))
    return root


def init_ros():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)


class ArmTopics:
    def __init__(self, node: Node):
        self.node = node
        self.joints = node.create_publisher(JointState, "joint_states", 10)
        self.targets = node.create_publisher(JointState, "joint_targets", 10)
        self.gripper = node.create_publisher(Float64, "gripper_position", 10)
        self.gripper_target = node.create_publisher(Float64, "gripper_target", 10)
        self.diagnostics = node.create_publisher(DiagnosticArray, "diagnostics", 10)

    def position(self, values, *, command=False):
        from so101_cartesian_control import ARM_JOINTS, ARM_OBSERVATION_KEYS, GRIPPER_KEY
        from so101_cartesian_control import lerobot_deg_to_model_rad

        if all(key in values for key in ARM_OBSERVATION_KEYS):
            angles = lerobot_deg_to_model_rad([values[key] for key in ARM_OBSERVATION_KEYS])
            if all(math.isfinite(angle) for angle in angles):
                message = JointState()
                message.header.stamp = self.node.get_clock().now().to_msg()
                message.name = list(ARM_JOINTS)
                message.position = angles.tolist()
                (self.targets if command else self.joints).publish(message)
        if GRIPPER_KEY in values and math.isfinite(values[GRIPPER_KEY]):
            (self.gripper_target if command else self.gripper).publish(
                Float64(data=float(values[GRIPPER_KEY]))
            )

    def diagnostic(self, name, values):
        status = DiagnosticStatus(name=name, hardware_id="so101")
        status.level = DiagnosticStatus.OK if values.get("online", True) else DiagnosticStatus.ERROR
        status.message = "OK" if status.level == DiagnosticStatus.OK else "No response"
        status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
        message = DiagnosticArray()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostics.publish(message)
