#!/usr/bin/env python3
"""Receive phone poses on Ubuntu and use the project's Cartesian arm controller."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2/so101_bringup/nodes"))
import so101_cartesian_control as cart

PERIOD = 0.05
MAX_SAMPLE_AGE = 0.25
MAX_CLUTCH_REACH_M = 0.055
TRANSLATION_SCALE = 0.5
MAX_SPEED_M_S = 0.020
MAX_JOINT_STEP_RAD = math.radians(1)
CAMERA_OFFSET_M = np.array([0.0, -0.02, 0.04])


@dataclass(frozen=True)
class Packet:
    sequence: int
    timestamp: float
    clutch: int
    position: np.ndarray
    orientation: Rotation
    b1: bool
    a3: float


def parse_packet(line):
    data = json.loads(line)
    sequence, clutch, b1 = data["sequence"], data["clutch"], data["b1"]
    timestamp, a3 = float(data["time"]), float(data["a3"])
    p = np.asarray(data["position"], dtype=float)
    q = np.asarray(data["orientation_wxyz"], dtype=float)
    if (
        type(sequence) is not int or sequence < 0
        or type(clutch) is not int or clutch < 0
        or type(b1) is not bool or (b1 and clutch == 0)
        or not math.isfinite(timestamp) or not -1.0 <= a3 <= 1.0
        or p.shape != (3,) or not np.all(np.isfinite(p))
        or q.shape != (4,) or not 0.9 < np.linalg.norm(q) < 1.1
    ):
        raise ValueError("invalid phone packet")
    orientation = Rotation.from_quat(np.roll(q, -1))
    return Packet(sequence, timestamp, clutch, p - orientation.apply(CAMERA_OFFSET_M),
                  orientation, b1, a3)


class PhoneStream:
    def __init__(self):
        self.latest = None
        self.closed = False
        self.accepted = 0
        self.rejected = 0
        self.peak_gap_ms = 0.0
        self.last_arrival = None
        self.thread = threading.Thread(target=self.read, daemon=True)

    def read(self):
        last_sequence = -1
        try:
            while line := sys.stdin.buffer.readline(2048):
                if not line.endswith(b"\n"):
                    raise ValueError("phone packet exceeds line limit")
                try:
                    packet = parse_packet(line)
                    if packet.sequence <= last_sequence:
                        raise ValueError("out-of-order phone packet")
                except (KeyError, TypeError, ValueError, OverflowError):
                    self.rejected += 1
                    self.latest = None
                    continue
                last_sequence = packet.sequence
                now = time.monotonic()
                if self.last_arrival is not None:
                    self.peak_gap_ms = max(self.peak_gap_ms, 1000 * (now - self.last_arrival))
                self.last_arrival = now
                self.latest = packet
                self.accepted += 1
        finally:
            self.latest = None
            self.closed = True

    def fresh_packet(self):
        packet = self.latest
        if packet is None:
            return None
        age = time.time() - packet.timestamp
        return packet if -0.10 <= age <= MAX_SAMPLE_AGE else None


def limited(vector, maximum):
    size = float(np.linalg.norm(vector))
    return vector * (maximum / size) if size > maximum else vector


class PhoneJog:
    def __init__(self, kinematics, initial_q, initial_grip, min_z_m):
        self.kin = kinematics
        self.q = initial_q.copy()
        self.grip = initial_grip
        self.min_z_m = min_z_m
        self.released = False
        self.clutch = None
        self.anchor = None
        self.phone_anchor = None
        self.basis = None
        self.last_packet = None
        self.last_phone_position = None
        self.last_goal_move = -math.inf

    def step(self, packet, measured_q, measured_grip, now):
        if packet is None or not packet.b1:
            if self.anchor is not None and now - self.last_goal_move < 0.20:
                self.q, self.grip = measured_q.copy(), measured_grip
            self.anchor = None
            # A stale input always requires a fresh B1 release before rearming.
            self.released = packet is not None
            return "hold" if packet is not None else "waiting for fresh phone input"
        if not self.released:
            return "release B1 to arm"
        if packet.sequence == self.last_packet:
            return None
        self.last_packet = packet.sequence
        if self.clutch != packet.clutch or self.anchor is None:
            if self.kin.joint_limit_violations(self.q, tolerance_deg=0.0):
                self.released = False
                return "outside working joint limits"
            self.clutch = packet.clutch
            self.anchor = self.kin.fk(self.q)
            self.phone_anchor = packet.position.copy()
            self.last_phone_position = packet.position.copy()
            self.basis = packet.orientation.inv()
            return "B1 engaged; reference captured"

        if np.linalg.norm(packet.position - self.last_phone_position) > 0.04:
            self.anchor = None
            self.released = False
            if now - self.last_goal_move < 0.20:
                self.q, self.grip = measured_q.copy(), measured_grip
            return "tracking jump; release B1"
        self.last_phone_position = packet.position.copy()
        displacement = self.basis.apply(packet.position - self.phone_anchor)
        dx, dy, dz = displacement
        delta = limited(TRANSLATION_SCALE * np.array([dy, -dx, dz]), MAX_CLUTCH_REACH_M)
        desired = self.anchor.xyz_m + delta
        desired[2] = max(self.min_z_m, desired[2])
        current = self.kin.fk(self.q)
        move = limited(desired - current.xyz_m, MAX_SPEED_M_S * PERIOD)

        if abs(packet.a3) >= 0.12:
            self.grip = float(np.clip(self.grip + packet.a3 * 20 * PERIOD, 0, 100))
            self.last_goal_move = now
        if np.linalg.norm(move) < 0.0005:
            return "tracking"
        target = cart.Pose(current.xyz_m + move, self.anchor.rotation)
        try:
            result = self.kin.ik(
                target, self.q, orientation_weight=0.05, max_iterations=36,
                position_tolerance_m=0.0003, orientation_tolerance_rad=math.radians(10),
            )
        except cart.CartesianControlError:
            return "workspace limit"
        if not result.success:
            return "IK hold"
        self.q += limited(result.q_arm_rad - self.q, MAX_JOINT_STEP_RAD)
        self.last_goal_move = now
        return "tracking"


def action_for(q, grip):
    return {
        **dict(zip(cart.ARM_OBSERVATION_KEYS, cart.model_rad_to_lerobot_deg(q), strict=True)),
        cart.GRIPPER_KEY: grip,
    }


def wait_for_released_phone(stream):
    end = time.monotonic() + 6
    while not stream.closed and time.monotonic() < end:
        packet = stream.fresh_packet()
        if packet is not None and not packet.b1:
            return
        time.sleep(0.05)
    raise RuntimeError("No fresh phone packet with B1 released. Check tracking and Mac/Ubuntu clocks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motion", action="store_true", help="Connect to the physical arm")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--min-z-mm", default=10., type=float)
    args = parser.parse_args()
    if not 0 <= args.min_z_mm <= 100:
        parser.error("--min-z-mm must be between 0 and 100")

    kin = cart.SO101Kinematics(cart.DEFAULT_URDF, cart.DEFAULT_CALIBRATION)
    neutral = json.loads((ROOT / "xbox_controller/neutral_pose.json").read_text())["joints_lerobot_deg"]
    q = cart.lerobot_deg_to_model_rad([neutral[name] for name in cart.ARM_JOINTS])
    gripper = float(neutral["gripper"])
    stream = PhoneStream()
    stream.thread.start()
    robot = node = rclpy = None
    jog = None
    observed = None
    try:
        if args.enable_motion:
            from common import ArmTopics, configure_project, init_ros
            import rclpy
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            configure_project(ROOT)
            init_ros()
            node = rclpy.create_node("phone_xyz", namespace="so101")
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.1)
            if any(full_name == "follower" and namespace == "/so101"
                   for full_name, namespace in node.get_node_names_and_namespaces()) or (
                subprocess.run(["fuser", args.port], capture_output=True).returncode == 0
            ):
                raise RuntimeError("Stop the Xbox/follower controller before connecting the phone.")
            wait_for_released_phone(stream)
            robot = SO101Follower(SO101FollowerConfig(
                port=args.port, id="my_follower", max_relative_target=3.,
                calibration_dir=ROOT / "config", disable_torque_on_disconnect=False,
            ))
            robot.connect(calibrate=False)
            observed = robot.get_observation()
            q = cart.lerobot_deg_to_model_rad([observed[name] for name in cart.ARM_OBSERVATION_KEYS])
            gripper = float(observed[cart.GRIPPER_KEY])
            kin.check_workspace(kin.fk(q).xyz_m)
            if kin.joint_limit_violations(q, tolerance_deg=0.0):
                raise RuntimeError("Start with the arm in the NEUTRAL working pose.")
            topics = ArmTopics(node)

        jog = PhoneJog(kin, q, gripper, args.min_z_mm / 1000.)
        print("LIVE phone XYZ" if robot is not None else "DRY RUN - no serial connection", flush=True)
        last_status = ""
        last_report = -math.inf
        deadline = time.monotonic()
        while not stream.closed:
            now = time.monotonic()
            if robot is not None:
                observed = robot.get_observation()
                topics.position(observed)
                q = cart.lerobot_deg_to_model_rad([observed[name] for name in cart.ARM_OBSERVATION_KEYS])
                gripper = float(observed[cart.GRIPPER_KEY])
            status = jog.step(stream.fresh_packet(), q, gripper, now)
            if status is not None and status != last_status:
                print(status, flush=True)
                last_status = status
            if robot is not None:
                sent = robot.send_action(action_for(jog.q, jog.grip))
                topics.position(sent, command=True)
                jog.q = cart.lerobot_deg_to_model_rad([sent[name] for name in cart.ARM_OBSERVATION_KEYS])
                jog.grip = float(sent[cart.GRIPPER_KEY])
                rclpy.spin_once(node, timeout_sec=0.0)
            else:
                q, gripper = jog.q.copy(), jog.grip
                if jog.anchor is not None and now - last_report >= 0.5:
                    xyz_mm = np.round(kin.fk(q).xyz_m * 1000, 1).tolist()
                    print(f"Dry target XYZ (mm): {xyz_mm}; gripper: {gripper:.1f}", flush=True)
                    last_report = now
            deadline = max(deadline + PERIOD, time.monotonic())
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        if robot is not None and robot.is_connected:
            try:
                if jog is not None and time.monotonic() - jog.last_goal_move < 0.20:
                    robot.send_action(robot.get_observation())
                print("Phone control stopped. Servos retain torque.", flush=True)
            finally:
                robot.disconnect()
        if node is not None:
            node.destroy_node()
            rclpy.try_shutdown()
        print(f"Ubuntu received {stream.accepted} valid packets; rejected {stream.rejected}; "
              f"longest receive gap {stream.peak_gap_ms:.1f} ms.", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        main()
    except KeyboardInterrupt:
        pass
