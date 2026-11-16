#!/usr/bin/env python3
"""Safety-first Cartesian control for a LeRobot SO-101 follower.

This single-file tool provides:
  * Forward kinematics (FK) from LeRobot joint angles.
  * Numerical inverse kinematics (IK) using damped least squares.
  * Cartesian straight-line planning.
  * Optional real-robot execution through LeRobot.

Motion commands are dry-run by default. Add --execute explicitly to move the
real robot. The gripper is not part of Cartesian IK and keeps its current value.

Important: SO-101 has five arm joints, so arbitrary 6-DoF poses are not always
reachable. Position is prioritized; orientation holding is optional and soft.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pinocchio as pin


DEFAULT_PORT = "/dev/cu.usbmodem5B3E0888091"
DEFAULT_ROBOT_ID = "my_follower"
DEFAULT_URDF = Path(__file__).resolve().parent / "simulation" / "so101_new_calib.urdf"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "config" / "my_follower.json"

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ARM_OBSERVATION_KEYS = tuple(f"{name}.pos" for name in ARM_JOINTS)
GRIPPER_KEY = "gripper.pos"
TOOL_FRAME = "gripper_frame_link"

# Adapter from LeRobot calibrated degrees to URDF radians.
# model_deg = lerobot_deg * JOINT_SCALE + JOINT_OFFSET_DEG
# These identity defaults must be verified against the physical arm before
# larger Cartesian movements. They are intentionally easy to tune in one place.
JOINT_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
JOINT_OFFSET_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


class CartesianControlError(RuntimeError):
    """Raised when a target cannot be planned or executed safely."""


@dataclass
class Pose:
    xyz_m: np.ndarray
    rotation: np.ndarray


@dataclass
class IKResult:
    success: bool
    q_arm_rad: np.ndarray
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    message: str


def lerobot_deg_to_model_rad(joints_deg: Sequence[float]) -> np.ndarray:
    values = np.asarray(joints_deg, dtype=float)
    if values.shape != (5,):
        raise ValueError(f"Expected five arm joints, got shape {values.shape}.")
    return np.deg2rad(values * JOINT_SCALE + JOINT_OFFSET_DEG)


def model_rad_to_lerobot_deg(q_arm_rad: Sequence[float]) -> np.ndarray:
    values = np.asarray(q_arm_rad, dtype=float)
    if values.shape != (5,):
        raise ValueError(f"Expected five arm joints, got shape {values.shape}.")
    return (np.rad2deg(values) - JOINT_OFFSET_DEG) / JOINT_SCALE


def rotation_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    return np.rad2deg(pin.rpy.matrixToRpy(rotation))


def format_vector(values: Sequence[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def print_pose(label: str, pose: Pose) -> None:
    print(label)
    print(f"  xyz mm:  {format_vector(pose.xyz_m * 1000.0, 2)}")
    print(f"  rpy deg: {format_vector(rotation_to_rpy_deg(pose.rotation), 2)}")


class SO101Kinematics:
    """URDF-backed SO-101 FK, Jacobian, IK, and Cartesian planning."""

    def __init__(
        self,
        urdf_path: Path,
        calibration_path: Path | None = None,
        joint_margin_deg: float = 3.0,
        tool_frame: str = TOOL_FRAME,
    ):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        available_frames = {frame.name for frame in self.model.frames}
        if tool_frame not in available_frames:
            raise ValueError(f"Tool frame '{tool_frame}' not found in URDF.")
        self.tool_frame = tool_frame
        self.tool_frame_id = self.model.getFrameId(tool_frame)

        self.q_indices: list[int] = []
        self.v_indices: list[int] = []
        for name in ARM_JOINTS:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Joint '{name}' not found in URDF.")
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Joint '{name}' is not a one-DoF joint.")
            self.q_indices.append(joint.idx_q)
            self.v_indices.append(joint.idx_v)

        self.q_neutral = pin.neutral(self.model)
        self.urdf_lower = self.model.lowerPositionLimit[self.q_indices].copy()
        self.urdf_upper = self.model.upperPositionLimit[self.q_indices].copy()
        self.hard_lower = self.urdf_lower.copy()
        self.hard_upper = self.urdf_upper.copy()
        self.lower = self.urdf_lower.copy()
        self.upper = self.urdf_upper.copy()
        self.limit_source = "URDF"

        if calibration_path is not None:
            calibration_path = Path(calibration_path).expanduser().resolve()
            if not calibration_path.is_file():
                raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
            calibration = json.loads(calibration_path.read_text())
            # STS3215 has 4096 positions per revolution (0..4095). In LeRobot
            # DEGREES mode, range_min/range_max define a symmetric calibrated
            # angle range around their midpoint.
            calibrated_half_ranges_deg = []
            for name in ARM_JOINTS:
                motor = calibration[name]
                span_ticks = float(motor["range_max"] - motor["range_min"])
                calibrated_half_ranges_deg.append(span_ticks * 180.0 / 4095.0)

            half_ranges = np.deg2rad(np.asarray(calibrated_half_ranges_deg, dtype=float))
            margin = math.radians(joint_margin_deg)
            if np.any(half_ranges <= margin):
                raise ValueError("Joint safety margin is larger than a calibrated joint range.")
            self.hard_lower = -half_ranges
            self.hard_upper = half_ranges
            self.lower = -half_ranges + margin
            self.upper = half_ranges - margin
            self.limit_source = f"calibration ({calibration_path.name}, {joint_margin_deg:.1f} deg margin)"

    def full_configuration(self, q_arm_rad: Sequence[float]) -> np.ndarray:
        q_arm = np.asarray(q_arm_rad, dtype=float)
        if q_arm.shape != (5,):
            raise ValueError(f"Expected five arm joints, got shape {q_arm.shape}.")
        q = self.q_neutral.copy()
        q[self.q_indices] = q_arm
        return q

    def fk(self, q_arm_rad: Sequence[float]) -> Pose:
        q = self.full_configuration(q_arm_rad)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.tool_frame_id]
        return Pose(placement.translation.copy(), placement.rotation.copy())

    def jacobian(self, q_arm_rad: Sequence[float]) -> np.ndarray:
        q = self.full_configuration(q_arm_rad)
        jacobian_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.tool_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return jacobian_full[:, self.v_indices]

    def joint_limit_violations(self, q_arm_rad: Sequence[float], tolerance_deg: float = 1.0) -> list[str]:
        q_arm = np.asarray(q_arm_rad, dtype=float)
        tolerance = math.radians(tolerance_deg)
        violations = []
        for index, name in enumerate(ARM_JOINTS):
            if q_arm[index] < self.lower[index] - tolerance or q_arm[index] > self.upper[index] + tolerance:
                violations.append(
                    f"{name}: {math.degrees(q_arm[index]):.2f} deg outside "
                    f"[{math.degrees(self.lower[index]):.2f}, {math.degrees(self.upper[index]):.2f}] deg "
                    f"from {self.limit_source}"
                )
        return violations

    def check_workspace(self, xyz_m: Sequence[float]) -> None:
        xyz = np.asarray(xyz_m, dtype=float)
        radius = float(np.linalg.norm(xyz))
        horizontal_radius = float(np.linalg.norm(xyz[:2]))
        if not np.all(np.isfinite(xyz)):
            raise CartesianControlError("Target contains NaN or infinity.")
        if radius > 0.50:
            raise CartesianControlError(f"Target radius {radius:.3f} m exceeds the 0.50 m safety envelope.")
        if horizontal_radius < 0.025 and xyz[2] < 0.08:
            raise CartesianControlError("Target is too close to the base at a low height.")
        if xyz[2] < -0.03 or xyz[2] > 0.48:
            raise CartesianControlError(f"Target z={xyz[2]:.3f} m is outside the safety envelope.")

    def ik(
        self,
        target: Pose,
        seed_q_arm_rad: Sequence[float],
        *,
        orientation_weight: float = 0.0,
        max_iterations: int = 250,
        position_tolerance_m: float = 0.001,
        orientation_tolerance_rad: float = math.radians(3.0),
        damping: float = 0.03,
        max_step_rad: float = math.radians(8.0),
    ) -> IKResult:
        self.check_workspace(target.xyz_m)
        if orientation_weight < 0.0 or orientation_weight > 1.0:
            raise ValueError("orientation_weight must be between 0 and 1.")

        q_arm = np.clip(np.asarray(seed_q_arm_rad, dtype=float), self.lower, self.upper)
        best_q = q_arm.copy()
        best_score = float("inf")
        last_pos_error = float("inf")
        last_rot_error = float("inf")

        for iteration in range(1, max_iterations + 1):
            current = self.fk(q_arm)
            position_error = target.xyz_m - current.xyz_m
            # World-aligned orientation error, compatible with the chosen Jacobian frame.
            rotation_error = pin.log3(target.rotation @ current.rotation.T)
            pos_norm = float(np.linalg.norm(position_error))
            rot_norm = float(np.linalg.norm(rotation_error))

            score = pos_norm + orientation_weight * rot_norm * 0.05
            if score < best_score:
                best_score = score
                best_q = q_arm.copy()
                last_pos_error = pos_norm
                last_rot_error = rot_norm

            orientation_ok = orientation_weight == 0.0 or rot_norm <= orientation_tolerance_rad
            if pos_norm <= position_tolerance_m and orientation_ok:
                return IKResult(True, q_arm, iteration, pos_norm, rot_norm, "IK converged.")

            jacobian = self.jacobian(q_arm)
            if orientation_weight > 0.0:
                error = np.concatenate((position_error, orientation_weight * rotation_error))
                weighted_jacobian = np.vstack((jacobian[:3, :], orientation_weight * jacobian[3:, :]))
            else:
                error = position_error
                weighted_jacobian = jacobian[:3, :]

            regularizer = (damping**2) * np.eye(weighted_jacobian.shape[0])
            try:
                delta_q = weighted_jacobian.T @ np.linalg.solve(
                    weighted_jacobian @ weighted_jacobian.T + regularizer,
                    error,
                )
            except np.linalg.LinAlgError:
                return IKResult(
                    False,
                    best_q,
                    iteration,
                    last_pos_error,
                    last_rot_error,
                    "IK linear solve failed near a singularity.",
                )

            largest_step = float(np.max(np.abs(delta_q)))
            if largest_step > max_step_rad:
                delta_q *= max_step_rad / largest_step

            q_arm = np.clip(q_arm + delta_q, self.lower, self.upper)

        return IKResult(
            False,
            best_q,
            max_iterations,
            last_pos_error,
            last_rot_error,
            "IK did not converge within the iteration limit.",
        )

    def plan_cartesian_line(
        self,
        start_q_arm_rad: Sequence[float],
        target: Pose,
        *,
        orientation_weight: float = 0.0,
        cartesian_step_m: float = 0.004,
    ) -> list[np.ndarray]:
        start_q = np.asarray(start_q_arm_rad, dtype=float)
        violations = self.joint_limit_violations(start_q)
        if violations:
            raise CartesianControlError(
                "Current pose is outside the configured safe joint limits:\n  " + "\n  ".join(violations)
            )

        start_pose = self.fk(start_q)
        distance = float(np.linalg.norm(target.xyz_m - start_pose.xyz_m))
        if distance > 0.03:
            raise CartesianControlError(
                f"Requested Cartesian move is {distance * 1000.0:.1f} mm; the per-command limit is 30 mm."
            )

        steps = max(1, int(math.ceil(distance / cartesian_step_m)))
        plan: list[np.ndarray] = []
        seed = start_q.copy()
        for step in range(1, steps + 1):
            alpha = step / steps
            waypoint = Pose(
                xyz_m=start_pose.xyz_m + alpha * (target.xyz_m - start_pose.xyz_m),
                rotation=target.rotation,
            )
            result = self.ik(waypoint, seed, orientation_weight=orientation_weight)
            if not result.success:
                raise CartesianControlError(
                    f"IK failed at waypoint {step}/{steps}: {result.message} "
                    f"position error={result.position_error_m * 1000.0:.2f} mm, "
                    f"orientation error={math.degrees(result.orientation_error_rad):.2f} deg"
                )
            if self.joint_limit_violations(result.q_arm_rad, tolerance_deg=0.0):
                raise CartesianControlError(f"IK waypoint {step}/{steps} violates a joint limit.")
            plan.append(result.q_arm_rad.copy())
            seed = result.q_arm_rad
        return plan

    def plan_joint_line(
        self,
        start_q_arm_rad: Sequence[float],
        target_q_arm_rad: Sequence[float],
        *,
        joint_step_rad: float = math.radians(1.5),
    ) -> list[np.ndarray]:
        """Interpolate a conservative joint-space path toward a safe target.

        The start may be inside the calibrated mechanical range but outside the
        configured safety margin. This permits recovery from a calibration-end
        pose, provided every joint moves monotonically toward the safe target.
        """
        start_q = np.asarray(start_q_arm_rad, dtype=float)
        target_q = np.asarray(target_q_arm_rad, dtype=float)
        if start_q.shape != (5,) or target_q.shape != (5,):
            raise ValueError("Joint-space planning expects five arm joints.")
        if not np.all(np.isfinite(start_q)) or not np.all(np.isfinite(target_q)):
            raise CartesianControlError("Joint-space target contains NaN or infinity.")
        if joint_step_rad <= 0.0:
            raise ValueError("joint_step_rad must be positive.")

        # Permit up to one degree of encoder/calibration rounding at the hard
        # range, but never initiate automatic recovery from a clearly invalid
        # reading or a joint beyond its calibrated mechanical range.
        hard_tolerance = math.radians(1.0)
        hard_bad = np.logical_or(
            start_q < self.hard_lower - hard_tolerance,
            start_q > self.hard_upper + hard_tolerance,
        )
        if np.any(hard_bad):
            details = []
            for index in np.flatnonzero(hard_bad):
                details.append(
                    f"{ARM_JOINTS[index]}={math.degrees(start_q[index]):.2f} deg, calibrated hard range "
                    f"[{math.degrees(self.hard_lower[index]):.2f}, "
                    f"{math.degrees(self.hard_upper[index]):.2f}] deg"
                )
            raise CartesianControlError(
                "Cannot automatically recover because a joint is beyond its calibrated range:\n  "
                + "\n  ".join(details)
            )

        target_violations = self.joint_limit_violations(target_q, tolerance_deg=0.0)
        if target_violations:
            raise CartesianControlError(
                "Joint-space target is outside the configured safe limits:\n  "
                + "\n  ".join(target_violations)
            )

        delta = target_q - start_q
        steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / joint_step_rad)))
        return [start_q + (step / steps) * delta for step in range(1, steps + 1)]


class SO101Driver:
    """Thin LeRobot wrapper; imported lazily so offline FK/IK stays usable."""

    def __init__(self, port: str, robot_id: str, max_relative_target_deg: float = 2.0):
        self.port = port
        self.robot_id = robot_id
        self.max_relative_target_deg = max_relative_target_deg
        self.robot = None

    def connect(self) -> None:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=self.port,
                id=self.robot_id,
                max_relative_target=self.max_relative_target_deg,
            )
        )
        self.robot.connect()
        # The USB serial bridge can need a brief settling period after the
        # configuration writes performed by SO101Follower.connect().
        time.sleep(0.15)

    def disconnect(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            self.robot.disconnect()

    def observation(self, attempts: int = 5, retry_delay_s: float = 0.05) -> dict[str, float]:
        if self.robot is None or not self.robot.is_connected:
            raise CartesianControlError("Robot is not connected.")
        if attempts < 1:
            raise ValueError("attempts must be at least 1.")

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                observation = self.robot.get_observation()
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(retry_delay_s * attempt)
            else:
                from so101_events import report_position

                return report_position(observation)

        raise CartesianControlError(
            f"Failed to read SO-101 observation after {attempts} attempts: {last_error}"
        ) from last_error

    def arm_q_model_rad(self, observation: dict[str, float]) -> np.ndarray:
        joints_deg = [float(observation[key]) for key in ARM_OBSERVATION_KEYS]
        return lerobot_deg_to_model_rad(joints_deg)

    def send_arm_q_model_rad(self, q_arm_rad: Sequence[float], gripper_value: float) -> dict[str, float]:
        if self.robot is None or not self.robot.is_connected:
            raise CartesianControlError("Robot is not connected.")
        joints_deg = model_rad_to_lerobot_deg(q_arm_rad)
        action = {key: float(value) for key, value in zip(ARM_OBSERVATION_KEYS, joints_deg, strict=True)}
        action[GRIPPER_KEY] = float(gripper_value)
        from so101_events import report_position

        return report_position(self.robot.send_action(action), command=True)


def print_joint_state(q_arm_rad: Sequence[float], label: str) -> None:
    model_deg = np.rad2deg(np.asarray(q_arm_rad, dtype=float))
    lerobot_deg = model_rad_to_lerobot_deg(q_arm_rad)
    print(label)
    for name, lr_deg, urdf_deg in zip(ARM_JOINTS, lerobot_deg, model_deg, strict=True):
        print(f"  {name:14s} LeRobot={lr_deg:8.2f} deg  URDF={urdf_deg:8.2f} deg")


def connect_driver(args: argparse.Namespace) -> SO101Driver:
    driver = SO101Driver(args.port, args.robot_id, args.max_relative_target_deg)
    print(f"Connecting to {args.port} with robot id '{args.robot_id}'...")
    driver.connect()
    return driver


def command_self_test(kinematics: SO101Kinematics, _args: argparse.Namespace) -> None:
    q_reference = np.deg2rad(np.array([15.0, -35.0, 45.0, 20.0, -10.0]))
    reference_pose = kinematics.fk(q_reference)
    print_pose("Reference FK pose", reference_pose)

    seed = q_reference + np.deg2rad(np.array([-8.0, 7.0, -6.0, 5.0, 4.0]))
    result = kinematics.ik(reference_pose, seed, orientation_weight=0.2)
    print(
        f"IK self-test: success={result.success}, iterations={result.iterations}, "
        f"position_error={result.position_error_m * 1000.0:.3f} mm, "
        f"orientation_error={math.degrees(result.orientation_error_rad):.3f} deg"
    )
    if not result.success:
        raise CartesianControlError(result.message)

    solved_pose = kinematics.fk(result.q_arm_rad)
    print_pose("Solved FK pose", solved_pose)
    print("Offline FK/IK self-test passed. No robot connection was made.")


def command_fk(kinematics: SO101Kinematics, args: argparse.Namespace) -> None:
    if args.joints_deg is not None:
        q_arm = lerobot_deg_to_model_rad(args.joints_deg)
        print_joint_state(q_arm, "Input joint state")
        print_pose("FK result", kinematics.fk(q_arm))
        return

    driver = connect_driver(args)
    try:
        observation = driver.observation()
        q_arm = driver.arm_q_model_rad(observation)
        print_joint_state(q_arm, "Current robot joint state")
        print_pose("Current FK result", kinematics.fk(q_arm))
        violations = kinematics.joint_limit_violations(q_arm)
        if violations:
            print("WARNING: current pose is outside the configured safe limits:")
            for violation in violations:
                print(f"  - {violation}")
    finally:
        driver.disconnect()


def execute_plan(
    driver: SO101Driver,
    kinematics: SO101Kinematics,
    plan: Sequence[np.ndarray],
    gripper_value: float,
    target: Pose,
    args: argparse.Namespace,
) -> None:
    print("\nMOTION ENABLED. Keep a hand near the power switch. Starting in 3 seconds...")
    for remaining in (3, 2, 1):
        print(f"  {remaining}")
        time.sleep(1.0)

    period = 1.0 / args.control_hz
    for index, q_waypoint in enumerate(plan, start=1):
        deadline = time.monotonic() + args.waypoint_timeout_s
        settled_reads = 0
        max_error_deg = float("inf")
        while time.monotonic() < deadline:
            # Re-send the waypoint until the physical joints have caught up.
            # LeRobot may clamp an individual command to max_relative_target;
            # repeated closed-loop sends safely walk toward the same waypoint.
            driver.send_arm_q_model_rad(q_waypoint, gripper_value)
            time.sleep(period)
            measured = driver.observation()
            measured_q = driver.arm_q_model_rad(measured)
            max_error_deg = float(np.max(np.abs(np.rad2deg(q_waypoint - measured_q))))
            if max_error_deg <= args.waypoint_tolerance_deg:
                settled_reads += 1
                if settled_reads >= 2:
                    break
            else:
                settled_reads = 0
        else:
            raise CartesianControlError(
                f"Waypoint {index}/{len(plan)} did not settle within "
                f"{args.waypoint_timeout_s:.1f} s; max joint error={max_error_deg:.2f} deg."
            )

        print(
            f"Executed waypoint {index}/{len(plan)} "
            f"(max joint error={max_error_deg:.2f} deg)"
        )

    time.sleep(0.5)
    final_observation = driver.observation()
    final_q = driver.arm_q_model_rad(final_observation)
    final_pose = kinematics.fk(final_q)
    final_error = float(np.linalg.norm(target.xyz_m - final_pose.xyz_m))
    print_joint_state(final_q, "Measured final joint state")
    print_pose("Measured final FK pose", final_pose)
    print(f"Measured Cartesian position error: {final_error * 1000.0:.2f} mm")
    if final_error > 0.005:
        raise CartesianControlError(
            f"Final Cartesian error {final_error * 1000.0:.2f} mm exceeds the 5 mm safety threshold."
        )


def wait_before_releasing_torque() -> None:
    print("\nHOLDING POSITION: motor torque remains enabled.")
    print("Support the arm with one hand, then press ENTER to release torque and disconnect.")
    try:
        input()
    except EOFError:
        print("No interactive input available; releasing torque now.")


def plan_and_maybe_execute(kinematics: SO101Kinematics, args: argparse.Namespace) -> None:
    driver = connect_driver(args)
    pose_holding = False
    release_prompt_done = False
    try:
        observation = driver.observation()
        start_q = driver.arm_q_model_rad(observation)
        start_pose = kinematics.fk(start_q)

        # Once an execution command has connected, explicitly command the
        # measured pose so the arm is supported while IK is being planned.
        if args.execute or args.hold_after:
            driver.send_arm_q_model_rad(start_q, float(observation[GRIPPER_KEY]))
            time.sleep(0.3)
            pose_holding = True
            print("Current pose captured. Motor torque is holding the arm; release your hand slowly if stable.")

        print_joint_state(start_q, "Current robot joint state")
        print_pose("Current FK pose", start_pose)

        if args.command == "move-relative":
            delta_m = np.array([args.dx_mm, args.dy_mm, args.dz_mm], dtype=float) / 1000.0
            target_xyz = start_pose.xyz_m + delta_m
        elif args.command == "move-to":
            target_xyz = np.array([args.x_mm, args.y_mm, args.z_mm], dtype=float) / 1000.0
        else:
            raise AssertionError(args.command)

        target = Pose(target_xyz, start_pose.rotation.copy())
        print_pose("Requested target pose", target)
        plan = kinematics.plan_cartesian_line(
            start_q,
            target,
            orientation_weight=args.orientation_weight,
            cartesian_step_m=args.cartesian_step_mm / 1000.0,
        )
        final_planned_pose = kinematics.fk(plan[-1])
        final_position_error = float(np.linalg.norm(target.xyz_m - final_planned_pose.xyz_m))
        print_joint_state(plan[-1], "Planned final joint state")
        print_pose("Planned final FK pose", final_planned_pose)
        print(f"Plan: {len(plan)} waypoints, predicted position error={final_position_error * 1000.0:.2f} mm")

        if not args.execute:
            print("DRY RUN complete: no motion command was sent. Add --execute only after checking this plan.")
            if args.hold_after:
                release_prompt_done = True
                wait_before_releasing_torque()
            return

        execute_plan(
            driver,
            kinematics,
            plan,
            float(observation[GRIPPER_KEY]),
            target,
            args,
        )
        if args.hold_after:
            release_prompt_done = True
            wait_before_releasing_torque()
    finally:
        if args.hold_after and pose_holding and not release_prompt_done:
            release_prompt_done = True
            wait_before_releasing_torque()
        driver.disconnect()


def print_console_help() -> None:
    print("Commands:")
    print("  status                 Show measured joints and FK pose")
    print("  home                   Move all five arm joints to the 0 deg neutral pose")
    print("  rel DX DY DZ           Move by millimeters in the base frame")
    print("  goto X Y Z             Move to an absolute base-frame position in millimeters")
    print("  help                   Show this help")
    print("  quit                   Support the arm, release torque, and exit")


def console_status(driver: SO101Driver, kinematics: SO101Kinematics) -> tuple[dict[str, float], np.ndarray, Pose]:
    observation = driver.observation()
    q_arm = driver.arm_q_model_rad(observation)
    pose = kinematics.fk(q_arm)
    print_joint_state(q_arm, "Measured joint state")
    print_pose("Measured FK pose", pose)
    violations = kinematics.joint_limit_violations(q_arm)
    if violations:
        print("WARNING: current pose is outside the configured safe limits:")
        for violation in violations:
            print(f"  - {violation}")
    return observation, q_arm, pose


def console_move_to(
    driver: SO101Driver,
    kinematics: SO101Kinematics,
    args: argparse.Namespace,
    target_xyz_m: np.ndarray,
) -> None:
    observation = driver.observation()
    start_q = driver.arm_q_model_rad(observation)
    start_pose = kinematics.fk(start_q)
    target = Pose(np.asarray(target_xyz_m, dtype=float), start_pose.rotation.copy())

    print_pose("Current FK pose", start_pose)
    print_pose("Requested target pose", target)
    plan = kinematics.plan_cartesian_line(
        start_q,
        target,
        orientation_weight=args.orientation_weight,
        cartesian_step_m=args.cartesian_step_mm / 1000.0,
    )
    planned_pose = kinematics.fk(plan[-1])
    predicted_error = float(np.linalg.norm(target.xyz_m - planned_pose.xyz_m))
    print_joint_state(plan[-1], "Planned final joint state")
    print_pose("Planned final FK pose", planned_pose)
    print(f"Plan: {len(plan)} waypoints, predicted position error={predicted_error * 1000.0:.2f} mm")

    confirmation = input("Type MOVE and press ENTER to execute, or press ENTER to cancel: ").strip()
    if confirmation != "MOVE":
        print("Motion cancelled. The current pose remains locked.")
        return

    execute_plan(
        driver,
        kinematics,
        plan,
        float(observation[GRIPPER_KEY]),
        target,
        args,
    )
    print("Motion complete. The final pose remains locked; enter another command when ready.")


def console_home(
    driver: SO101Driver,
    kinematics: SO101Kinematics,
    args: argparse.Namespace,
    *,
    automatic: bool = False,
) -> None:
    """Move all arm joints to calibrated 0 degrees and keep holding there."""
    observation = driver.observation()
    start_q = driver.arm_q_model_rad(observation)
    neutral_q = lerobot_deg_to_model_rad([0.0, 0.0, 0.0, 0.0, 0.0])
    neutral_pose = kinematics.fk(neutral_q)
    plan = kinematics.plan_joint_line(
        start_q,
        neutral_q,
        joint_step_rad=math.radians(args.home_joint_step_deg),
    )

    print_joint_state(start_q, "Home start joint state")
    print_joint_state(neutral_q, "Neutral target joint state")
    print_pose("Neutral target FK pose", neutral_pose)
    print(
        f"Home plan: {len(plan)} joint-space waypoints, "
        f"maximum step={args.home_joint_step_deg:.2f} deg"
    )
    print("Visually check that the swept path is clear of the table, cables, and other objects.")

    if not automatic:
        confirmation = input("Type HOME and press ENTER to execute, or press ENTER to cancel: ").strip()
        if confirmation != "HOME":
            print("Home motion cancelled. The current pose remains locked.")
            return

    execute_plan(
        driver,
        kinematics,
        plan,
        float(observation[GRIPPER_KEY]),
        neutral_pose,
        args,
    )
    print("NEUTRAL POSE REACHED. The arm remains locked at the 0 deg joint pose.")


def command_console(kinematics: SO101Kinematics, args: argparse.Namespace) -> None:
    """Persistent Cartesian controller that keeps motor torque enabled."""
    driver = connect_driver(args)
    release_confirmed = False
    try:
        observation = driver.observation()
        current_q = driver.arm_q_model_rad(observation)
        driver.send_arm_q_model_rad(current_q, float(observation[GRIPPER_KEY]))
        time.sleep(0.3)
        print("\nCURRENT POSE LOCKED: the servo position loops are holding the arm against gravity.")
        print("Keep a hand near the power switch and do not leave the arm unattended.")
        console_status(driver, kinematics)
        if args.home_on_start:
            print("\nHOME-ON-START ENABLED: automatic neutral motion will begin after the safety countdown.")
            console_home(driver, kinematics, args, automatic=True)
        print_console_help()

        while True:
            try:
                raw = input("\nso101> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExit requested.")
                wait_before_releasing_torque()
                release_confirmed = True
                break

            if not raw:
                continue
            parts = raw.split()
            command = parts[0].lower()

            if command in {"quit", "exit"}:
                wait_before_releasing_torque()
                release_confirmed = True
                break
            if command in {"help", "?"}:
                print_console_help()
                continue
            if command in {"status", "where"}:
                console_status(driver, kinematics)
                continue
            if command == "home":
                try:
                    console_home(driver, kinematics, args)
                except Exception as exc:
                    print(f"COMMAND ERROR: {exc}")
                    print("The controller remains connected and keeps the current goal position.")
                continue

            try:
                if command == "rel":
                    if len(parts) != 4:
                        raise ValueError("Usage: rel DX DY DZ  (millimeters)")
                    delta_m = np.asarray([float(value) for value in parts[1:]], dtype=float) / 1000.0
                    _, _, current_pose = console_status(driver, kinematics)
                    console_move_to(driver, kinematics, args, current_pose.xyz_m + delta_m)
                elif command == "goto":
                    if len(parts) != 4:
                        raise ValueError("Usage: goto X Y Z  (millimeters)")
                    target_m = np.asarray([float(value) for value in parts[1:]], dtype=float) / 1000.0
                    console_move_to(driver, kinematics, args, target_m)
                else:
                    print(f"Unknown command: {command}")
                    print_console_help()
            except Exception as exc:
                print(f"COMMAND ERROR: {exc}")
                print("The controller remains connected and keeps the current goal position.")
    finally:
        if not release_confirmed and driver.robot is not None and driver.robot.is_connected:
            wait_before_releasing_torque()
        driver.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 Cartesian FK/IK controller. Motion commands default to dry-run.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF path. Default: {DEFAULT_URDF}")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=f"LeRobot calibration JSON used for hardware joint limits. Default: {DEFAULT_CALIBRATION}",
    )
    parser.add_argument(
        "--joint-margin-deg",
        type=float,
        default=3.0,
        help="Safety margin removed from both ends of each calibrated range. Default: 3 deg.",
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Robot serial port. Default: {DEFAULT_PORT}")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help=f"LeRobot calibration id. Default: {DEFAULT_ROBOT_ID}")
    parser.add_argument(
        "--max-relative-target-deg",
        type=float,
        default=2.0,
        help="LeRobot per-command joint safety limit in degrees. Default: 2",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run offline FK/IK consistency checks; never connects to robot.")

    fk_parser = subparsers.add_parser("fk", help="Compute FK from explicit joints or the connected robot.")
    fk_parser.add_argument(
        "--joints-deg",
        nargs=5,
        type=float,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST_FLEX", "WRIST_ROLL"),
        help="Five LeRobot joint angles in degrees. Omit to read the real robot.",
    )

    def add_controller_tuning_arguments(motion_parser: argparse.ArgumentParser) -> None:
        motion_parser.add_argument(
            "--orientation-weight",
            type=float,
            default=0.0,
            help="Soft orientation-holding weight from 0 to 1. Default: 0 (position only).",
        )
        motion_parser.add_argument(
            "--cartesian-step-mm",
            type=float,
            default=2.0,
            help="Cartesian distance between IK waypoints. Default: 2 mm.",
        )
        motion_parser.add_argument("--control-hz", type=float, default=15.0, help="Execution rate. Default: 15 Hz.")
        motion_parser.add_argument(
            "--waypoint-tolerance-deg",
            type=float,
            default=0.5,
            help="Maximum joint error before advancing to the next waypoint. Default: 0.5 deg.",
        )
        motion_parser.add_argument(
            "--waypoint-timeout-s",
            type=float,
            default=3.0,
            help="Maximum settling time for each waypoint. Default: 3 s.",
        )

    def add_motion_arguments(motion_parser: argparse.ArgumentParser) -> None:
        add_controller_tuning_arguments(motion_parser)
        motion_parser.add_argument(
            "--execute",
            action="store_true",
            help="Actually move the robot. Omit for planning-only dry-run.",
        )
        motion_parser.add_argument(
            "--hold-after",
            action="store_true",
            help="Keep motor torque enabled after planning/motion until ENTER is pressed.",
        )

    relative = subparsers.add_parser("move-relative", help="Plan/move by a relative Cartesian offset.")
    relative.add_argument("--dx-mm", type=float, default=0.0)
    relative.add_argument("--dy-mm", type=float, default=0.0)
    relative.add_argument("--dz-mm", type=float, default=0.0)
    add_motion_arguments(relative)

    absolute = subparsers.add_parser("move-to", help="Plan/move to an absolute base-frame position.")
    absolute.add_argument("--x-mm", type=float, required=True)
    absolute.add_argument("--y-mm", type=float, required=True)
    absolute.add_argument("--z-mm", type=float, required=True)
    add_motion_arguments(absolute)

    console = subparsers.add_parser(
        "console",
        help="Start a persistent FK/IK controller that keeps the current pose locked.",
    )
    add_controller_tuning_arguments(console)
    console.add_argument(
        "--home-on-start",
        action="store_true",
        help="After locking the current pose, automatically move all arm joints to calibrated 0 deg.",
    )
    console.add_argument(
        "--home-joint-step-deg",
        type=float,
        default=1.5,
        help="Maximum joint-space interpolation step while moving home. Default: 1.5 deg.",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.max_relative_target_deg <= 0.0 or args.max_relative_target_deg > 5.0:
        raise CartesianControlError("--max-relative-target-deg must be within (0, 5].")
    if args.joint_margin_deg < 1.0 or args.joint_margin_deg > 10.0:
        raise CartesianControlError("--joint-margin-deg must be within [1, 10].")
    if args.command in {"move-relative", "move-to", "console"}:
        if args.cartesian_step_mm <= 0.0 or args.cartesian_step_mm > 10.0:
            raise CartesianControlError("--cartesian-step-mm must be within (0, 10].")
        if args.control_hz <= 0.0 or args.control_hz > 30.0:
            raise CartesianControlError("--control-hz must be within (0, 30].")
        if args.waypoint_tolerance_deg < 0.1 or args.waypoint_tolerance_deg > 2.0:
            raise CartesianControlError("--waypoint-tolerance-deg must be within [0.1, 2].")
        if args.waypoint_timeout_s < 0.5 or args.waypoint_timeout_s > 10.0:
            raise CartesianControlError("--waypoint-timeout-s must be within [0.5, 10].")
        if args.orientation_weight < 0.0 or args.orientation_weight > 1.0:
            raise CartesianControlError("--orientation-weight must be between 0 and 1.")
    if args.command == "console":
        if args.home_joint_step_deg < 0.25 or args.home_joint_step_deg > 2.0:
            raise CartesianControlError("--home-joint-step-deg must be within [0.25, 2.0].")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_cli(args)
        kinematics = SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
        if args.command == "self-test":
            command_self_test(kinematics, args)
        elif args.command == "fk":
            command_fk(kinematics, args)
        elif args.command in {"move-relative", "move-to"}:
            plan_and_maybe_execute(kinematics, args)
        elif args.command == "console":
            command_console(kinematics, args)
        else:
            parser.error(f"Unknown command: {args.command}")
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
