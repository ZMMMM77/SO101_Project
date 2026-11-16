"""Bring up two cameras and Xbox teleoperation on the Ubuntu USB host."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        "project_dir": os.getcwd(),
        "port": "/dev/ttyACM0",
        "robot_id": "my_follower",
        "wrist_device": "/dev/v4l/by-path/pci-0000:02:03.0-usb-0:1:1.0-video-index0",
        "top_device": "/dev/v4l/by-path/pci-0000:02:03.0-usb-0:2:1.0-video-index0",
        "cameras": "true",
        "xbox": "true",
        "controller_index": "0",
        "enable_motion": "false",
        "follower": "true",
    }
    arguments = [DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()]
    project = {"project_dir": LaunchConfiguration("project_dir")}
    nodes = [
        Node(
            package="so101_bringup", executable="camera", namespace="so101",
            output="screen", condition=IfCondition(LaunchConfiguration("cameras")),
            parameters=[{f"{view}_device": LaunchConfiguration(f"{view}_device")
                         for view in ("wrist", "top")}],
        ),
        Node(
            package="so101_bringup", executable="xbox_input", namespace="so101",
            output="screen", condition=IfCondition(LaunchConfiguration("xbox")),
            parameters=[project, {"controller_index": ParameterValue(
                LaunchConfiguration("controller_index"), value_type=int
            )}],
        ),
        Node(
            package="so101_bringup", executable="follower", namespace="so101",
            output="screen", condition=IfCondition(LaunchConfiguration("follower")),
            parameters=[project, {
                "port": LaunchConfiguration("port"), "robot_id": LaunchConfiguration("robot_id"),
                "enable_motion": ParameterValue(LaunchConfiguration("enable_motion"), value_type=bool),
            }],
        ),
    ]
    return LaunchDescription(arguments + nodes)
