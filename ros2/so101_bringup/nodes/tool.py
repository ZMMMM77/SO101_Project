#!/usr/bin/env python3
"""ROS entry point for the existing interactive SO-101 operations."""

import argparse
import importlib.util
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from common import ArmTopics, configure_project, init_ros

TOOLS = {
    "cartesian": ("so101_cartesian_control.py", "front"),
    "freedrive": ("so101_freedrive_sdk.py", "front"),
    "iphone": ("scripts/ios_phone_so101_joint_teleop.py", "end"),
    "scan": ("scripts/so101_bus_scan.py", "end"),
    "joint-test": ("scripts/so101_follower_test.py", "end"),
    "joint-test-basic": ("so101_follower_test.py", "end"),
    "xbox": ("xbox_controller/xbox_official_control.py", "end"),
    "xbox-legacy": ("xbox_controller/xbox_so101_control.py", "end"),
}


def load_tool(path):
    spec = importlib.util.spec_from_file_location("so101_selected_tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_tool(root, name, args, topics):
    from so101_events import observe_with

    if name == "phone-discovery":
        import hebi

        lookup = hebi.Lookup()
        print("Scanning for HEBI Mobile I/O for 10 seconds...", flush=True)
        time.sleep(10)
        print(lookup.entrylist)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        topics.diagnostic("phone", {"online": group is not None})
        if group is None:
            raise RuntimeError("HEBI / mobileIO was not found")
        print(f"Mobile I/O found: group size={group.size}")
        return 0

    filename, port_position = TOOLS[name]
    needs_port = name != "xbox-legacy" or any(arg in {"auto-test", "control"} for arg in args)
    if needs_port and not any(arg == "--port" or arg.startswith("--port=") for arg in args):
        port = ["--port", "/dev/ttyACM0"]
        args = port + args if port_position == "front" else args + port
    sys.argv = [filename, *args]
    module = load_tool(root / filename)
    with observe_with(topics.position):
        if name == "scan":
            return module.main(on_result=topics.diagnostic)
        if name in {"iphone", "joint-test", "joint-test-basic"}:
            from so101_events import report_position

            class ObservedFollower(module.SO101Follower):
                def get_observation(self):
                    return report_position(super().get_observation())

                def send_action(self, action):
                    return report_position(super().send_action(action), command=True)

            return module.main(robot_factory=ObservedFollower)
        return module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("tool", choices=[*TOOLS, "phone-discovery"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args(remove_ros_args(sys.argv)[1:])
    init_ros()
    node = Node(options.tool.replace("-", "_"), namespace="so101")
    try:
        root = configure_project(options.project_dir)
        topics = ArmTopics(node)
        rclpy.spin_once(node, timeout_sec=0.25)
        result = run_tool(root, options.tool, options.arguments, topics)
        rclpy.spin_once(node, timeout_sec=0.25)
        return result or 0
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
