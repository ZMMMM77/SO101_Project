#!/usr/bin/env python3
"""Safety-first iPhone HEBI Mobile I/O teleop for a SO-101 follower.

This is a conservative joint-space bridge for first bench tests. It does not
try to solve full end-effector IK. Hold B1 on the phone to enable motion.
Without --enable-motion it only prints the targets it would send.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import IOSPhone


DEFAULT_PORT = "/dev/cu.usbmodem5B3E0888091"
DEFAULT_ID = "my_follower"

ARM_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
)
ALL_JOINTS = (*ARM_JOINTS, "gripper.pos")
DRY_RUN_ANCHOR = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_joint(joint: str, value: float) -> float:
    if joint == "gripper.pos":
        return clamp(value, 0.0, 100.0)
    return clamp(value, -180.0, 180.0)


def bounded_offset(value: float, max_abs: float) -> float:
    return clamp(value, -max_abs, max_abs)


def compact(values: dict[str, float]) -> str:
    return " ".join(f"{joint.removesuffix('.pos')}={values[joint]:6.1f}" for joint in ALL_JOINTS)


def build_target(
    anchor: dict[str, float],
    action: dict[str, Any],
    gripper_target: float,
    dt_s: float,
    linear_scale_deg_per_m: float,
    rot_scale_deg_per_rad: float,
    max_joint_offset: float,
    gripper_speed_per_s: float,
    deadband: float,
) -> tuple[dict[str, float], float]:
    pos = np.asarray(action["phone.pos"], dtype=float)
    rotvec = np.asarray(action["phone.rot"].as_rotvec(), dtype=float)
    raw_inputs = action.get("phone.raw_inputs", {})

    # Conservative direct mapping for bench validation, not production IK.
    offsets = {
        "shoulder_pan.pos": bounded_offset(pos[0] * linear_scale_deg_per_m, max_joint_offset),
        "shoulder_lift.pos": bounded_offset(pos[2] * linear_scale_deg_per_m, max_joint_offset),
        "elbow_flex.pos": bounded_offset(-pos[1] * linear_scale_deg_per_m, max_joint_offset),
        "wrist_flex.pos": bounded_offset(rotvec[0] * rot_scale_deg_per_rad, max_joint_offset),
        "wrist_roll.pos": bounded_offset(-rotvec[2] * rot_scale_deg_per_rad, max_joint_offset),
    }

    a3 = float(raw_inputs.get("a3", 0.0))
    if abs(a3) < deadband:
        a3 = 0.0
    gripper_target = clamp_joint("gripper.pos", gripper_target + a3 * gripper_speed_per_s * dt_s)

    target = dict(anchor)
    for joint, offset in offsets.items():
        target[joint] = clamp_joint(joint, float(anchor[joint]) + offset)
    target["gripper.pos"] = gripper_target
    return target, gripper_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use iPhone HEBI Mobile I/O as a conservative SO-101 follower teleop input."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"SO-101 USB serial port. Default: {DEFAULT_PORT}")
    parser.add_argument("--id", default=DEFAULT_ID, help=f"LeRobot calibration id. Default: {DEFAULT_ID}")
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Actually send joint targets to the robot. Omit this for dry-run target printing.",
    )
    parser.add_argument(
        "--connect-robot-in-dry-run",
        action="store_true",
        help="Connect to the SO-101 even without --enable-motion, to anchor dry-run targets to the real pose.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Control loop rate. Default: 10 Hz")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Run time in seconds after calibration. Use 0 for no time limit. Default: 30",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=90.0,
        help="Degrees of joint offset per meter of phone translation. Default: 90",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=12.0,
        help="Degrees of wrist offset per radian of phone rotation. Default: 12",
    )
    parser.add_argument(
        "--max-joint-offset",
        type=float,
        default=10.0,
        help="Maximum commanded offset from the B1 press anchor for each arm joint. Default: 10 deg",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=3.0,
        help="LeRobot per-command relative target safety limit. Default: 3",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=20.0,
        help="Gripper target units per second at full A3 input. Default: 20",
    )
    parser.add_argument("--deadband", type=float, default=0.05, help="Analog input deadband. Default: 0.05")
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Seconds between status lines. Default: 0.5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    period_s = 1.0 / args.fps

    print("iPhone SO-101 teleop wrapper")
    print(f"  robot port: {args.port}")
    print(f"  robot id:   {args.id}")
    print(f"  mode:       {'MOTION ENABLED' if args.enable_motion else 'DRY RUN - no robot motion'}")
    print("\nPhone setup:")
    print("  1. Keep HEBI Mobile I/O open on the iPhone.")
    print("  2. Hold the phone top edge forward and screen up.")
    print("  3. Press and hold B1 once when prompted to calibrate.")
    print("  4. During the loop, hold B1 to enable targets; release B1 to freeze.")
    print("  5. A3 controls the gripper target.")

    phone = IOSPhone(PhoneConfig(phone_os=PhoneOS.IOS))
    robot = None
    should_connect_robot = args.enable_motion or args.connect_robot_in_dry_run
    if should_connect_robot:
        robot = SO101Follower(
            SO101FollowerConfig(
                port=args.port,
                id=args.id,
                max_relative_target=args.max_relative_target,
            )
        )

    try:
        phone.connect()
        if should_connect_robot:
            assert robot is not None
            try:
                robot.connect()
            except RuntimeError as exc:
                if "Input voltage error" in str(exc):
                    raise SystemExit(
                        "SO-101 reported an input voltage error while connecting. "
                        "Check the arm power supply/battery and the servo power wiring before enabling motion."
                    ) from exc
                raise
            anchor = robot.get_observation()
        else:
            anchor = dict(DRY_RUN_ANCHOR)
        gripper_target = float(anchor["gripper.pos"])
        if should_connect_robot:
            print(f"\nRobot connected. Current: {compact(anchor)}")
        else:
            print(f"\nPhone-only dry run. Synthetic anchor: {compact(anchor)}")
        print("Entering control loop. Press Ctrl-C to stop.")

        start_s = time.monotonic()
        last_s = start_s
        last_print_s = 0.0
        was_enabled = False

        while args.duration <= 0 or time.monotonic() - start_s < args.duration:
            loop_s = time.monotonic()
            dt_s = max(1e-3, loop_s - last_s)
            last_s = loop_s

            action = phone.get_action()
            enabled = bool(action.get("phone.enabled", False))

            if not enabled:
                if was_enabled:
                    if should_connect_robot:
                        assert robot is not None
                        anchor = robot.get_observation()
                    else:
                        anchor = dict(DRY_RUN_ANCHOR)
                    gripper_target = float(anchor["gripper.pos"])
                    print("B1 released. Motion frozen; anchor refreshed.")
                was_enabled = False
                time.sleep(period_s)
                continue

            if enabled and not was_enabled:
                if should_connect_robot:
                    assert robot is not None
                    anchor = robot.get_observation()
                else:
                    anchor = dict(DRY_RUN_ANCHOR)
                gripper_target = float(anchor["gripper.pos"])
                print("B1 held. Motion targets enabled from current robot pose.")
                was_enabled = True

            target, gripper_target = build_target(
                anchor=anchor,
                action=action,
                gripper_target=gripper_target,
                dt_s=dt_s,
                linear_scale_deg_per_m=args.linear_scale,
                rot_scale_deg_per_rad=args.rot_scale,
                max_joint_offset=args.max_joint_offset,
                gripper_speed_per_s=args.gripper_speed,
                deadband=args.deadband,
            )

            sent = target
            if args.enable_motion:
                assert robot is not None
                sent = robot.send_action(target)

            now_s = time.monotonic()
            if now_s - last_print_s >= args.print_every:
                label = "sent" if args.enable_motion else "dry"
                print(f"{label}: {compact(sent)}")
                last_print_s = now_s

            sleep_s = period_s - (time.monotonic() - loop_s)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        if phone.is_connected:
            phone.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
