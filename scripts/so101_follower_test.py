#!/usr/bin/env python3
"""Small safety-first test script for a calibrated SO-101 follower arm."""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


DEFAULT_PORT = "/dev/tty.usbmodem5B3E0888091"
DEFAULT_ID = "my_follower"
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def print_observation(title: str, observation: dict[str, float]) -> None:
    print(f"\n{title}")
    for key in sorted(observation):
        value = observation[key]
        if isinstance(value, float):
            print(f"  {key:18s} {value:8.2f}")
        else:
            print(f"  {key:18s} {value}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_joint_target(observation: dict[str, float], joint: str, delta: float) -> dict[str, float]:
    target = dict(observation)
    current = float(observation[joint])
    if joint == "gripper.pos":
        target[joint] = clamp(current + delta, 0.0, 100.0)
    else:
        target[joint] = clamp(current + delta, -180.0, 180.0)
    return target


def delta_toward_center(joint: str, current: float, magnitude: float) -> float:
    """Choose a visible test direction away from the nearest calibrated edge."""

    midpoint = 50.0 if joint == "gripper.pos" else 0.0
    return -abs(magnitude) if current > midpoint else abs(magnitude)


def move_and_report(robot: SO101Follower, title: str, target: dict[str, float], pause_s: float) -> None:
    sent = robot.send_action(target)
    print_observation(title, sent)
    time.sleep(pause_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect to a calibrated SO-101 follower and optionally test small motion on each joint."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"MotorBus serial port. Default: {DEFAULT_PORT}")
    parser.add_argument("--id", default=DEFAULT_ID, help=f"Calibration id. Default: {DEFAULT_ID}")
    parser.add_argument(
        "--move-gripper",
        action="store_true",
        help="Move only the gripper by a small amount, then return it near the starting position.",
    )
    parser.add_argument(
        "--test-all-joints",
        action="store_true",
        help="Move each joint by a small amount, return it, then continue to the next joint.",
    )
    parser.add_argument(
        "--joint",
        choices=JOINTS,
        help="Move only one joint by a small amount, then return it near the starting position.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=8.0,
        help="Small movement amount. Degrees for arm joints, 0-100 units for gripper. Default: 8.0",
    )
    parser.add_argument(
        "--toward-center",
        action="store_true",
        help="For each selected joint, choose the delta sign that moves toward its calibrated midpoint.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.8,
        help="Seconds to wait after each commanded move. Default: 0.8",
    )
    args = parser.parse_args()

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        max_relative_target=10.0,
    )
    robot = SO101Follower(config)

    try:
        robot.connect()
        observation = robot.get_observation()
        print_observation("Current position", observation)

        if not args.move_gripper and not args.test_all_joints and not args.joint:
            print(
                "\nRead-only check complete. Add --move-gripper, --joint, or --test-all-joints to run small motions."
            )
            return

        if args.move_gripper:
            print("\nMoving gripper by a small amount. Keep a hand near power in case you need to cut it.")
            first_target = make_joint_target(observation, "gripper.pos", args.delta)
            move_and_report(robot, "Sent gripper target +delta", first_target, args.pause)

            second_observation = robot.get_observation()
            second_target = make_joint_target(second_observation, "gripper.pos", -args.delta)
            move_and_report(robot, "Sent gripper target -delta", second_target, args.pause)

        if args.test_all_joints:
            print("\nTesting each joint by a small amount. Keep a hand near power in case you need to cut it.")
            for joint in JOINTS:
                print(f"\nTesting {joint}")
                start_observation = robot.get_observation()
                start_position = float(start_observation[joint])

                test_delta = (
                    delta_toward_center(joint, start_position, args.delta)
                    if args.toward_center
                    else args.delta
                )
                positive_target = make_joint_target(start_observation, joint, test_delta)
                move_and_report(robot, f"Sent {joint} delta={test_delta:+g}", positive_target, args.pause)

                return_observation = robot.get_observation()
                return_target = dict(return_observation)
                return_target[joint] = start_position
                move_and_report(robot, f"Returned {joint}", return_target, args.pause)

        if args.joint:
            print(f"\nTesting {args.joint}. Keep a hand near power in case you need to cut it.")
            start_observation = robot.get_observation()
            start_position = float(start_observation[args.joint])

            test_delta = (
                delta_toward_center(args.joint, start_position, args.delta)
                if args.toward_center
                else args.delta
            )
            positive_target = make_joint_target(start_observation, args.joint, test_delta)
            move_and_report(robot, f"Sent {args.joint} delta={test_delta:+g}", positive_target, args.pause)

            return_observation = robot.get_observation()
            return_target = dict(return_observation)
            return_target[args.joint] = start_position
            move_and_report(robot, f"Returned {args.joint}", return_target, args.pause)

        final_observation = robot.get_observation()
        print_observation("Final position", final_observation)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
