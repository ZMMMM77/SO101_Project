#!/usr/bin/env python3
"""Safety-first Xbox Series X|S control for an SO-101 follower.

Subcommands:
  self-test   Offline mapper + FK/IK validation. No controller or robot.
  input-test  Read and display Xbox controls. No robot.
  control     Real-time Cartesian control. Requires --execute to connect.

This tool intentionally reuses the existing SO101_Project kinematics and
LeRobot driver instead of modifying the installed LeRobot package.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from original_pose_config import ORIGINAL_POSE_LEROBOT_DEG
from xbox_teleoperator import GripperAction, XboxInputError, XboxLeRobotTeleop, XboxState


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = LOCAL_PROJECT_ROOT

UNFOLD_JOINT_ORDER = ("wrist_flex", "elbow_flex", "shoulder_lift")
UNFOLD_TARGET_DEG = 0.0
UNFOLD_DONE_TOLERANCE_DEG = 5.0


class XboxControlError(RuntimeError):
    """Raised when a safety condition prevents controller operation."""


def load_cartesian_module(project_root: Path) -> ModuleType:
    project_root = project_root.expanduser().resolve()
    module_path = project_root / "so101_cartesian_control.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Cartesian controller not found: {module_path}")
    sys.path.insert(0, str(project_root))
    return importlib.import_module("so101_cartesian_control")


def controller_neutral(state: XboxState, tolerance: float = 0.20) -> bool:
    sticks = (state.left_x, state.left_y, state.right_x, state.right_y)
    return (
        max(abs(value) for value in sticks) <= tolerance
        and state.left_trigger <= tolerance
        and state.right_trigger <= tolerance
        and not state.rb
    )


def button_names(state: XboxState) -> str:
    pressed = [
        name.upper()
        for name in ("a", "b", "x", "y", "lb", "rb", "view", "menu", "left_stick", "right_stick")
        if getattr(state, name)
    ]
    return ",".join(pressed) if pressed else "-"


def state_line(state: XboxState, action: dict[str, float | int]) -> str:
    return (
        f"L=({state.left_x:+.2f},{state.left_y:+.2f}) "
        f"R=({state.right_x:+.2f},{state.right_y:+.2f}) "
        f"LT={state.left_trigger:.2f} RT={state.right_trigger:.2f} "
        f"buttons={button_names(state):<12s} "
        f"LeRobot=({float(action['delta_x']):+.2f},"
        f"{float(action['delta_y']):+.2f},{float(action['delta_z']):+.2f},"
        f"wr={float(action['delta_wrist_roll']):+.2f},"
        f"g={int(action['gripper'])})"
    )


def compact_joint_line(cart: ModuleType, measured_q: np.ndarray, target_q: np.ndarray) -> str:
    measured_deg = cart.model_rad_to_lerobot_deg(measured_q)
    target_deg = cart.model_rad_to_lerobot_deg(target_q)
    labels = ("P", "S", "E", "WF", "WR")
    measured = ",".join(f"{label}={value:+.1f}" for label, value in zip(labels, measured_deg, strict=True))
    target = ",".join(f"{label}={value:+.1f}" for label, value in zip(labels, target_deg, strict=True))
    return f"q=[{measured}] target=[{target}]"


def make_teleop(args: argparse.Namespace) -> XboxLeRobotTeleop:
    return XboxLeRobotTeleop(
        controller_index=args.controller_index,
        deadzone=args.deadzone,
        trigger_threshold=args.trigger_threshold,
        require_xbox_name=not args.allow_non_xbox,
    )


def state_from_sent_action(
    cart: ModuleType,
    sent_action: dict[str, float],
) -> tuple[np.ndarray, float]:
    """Convert LeRobot's actually transmitted (possibly clamped) action to command state."""

    arm_deg = np.asarray(
        [float(sent_action[key]) for key in cart.ARM_OBSERVATION_KEYS],
        dtype=float,
    )
    gripper = float(sent_action[cart.GRIPPER_KEY])
    if not np.all(np.isfinite(arm_deg)) or not math.isfinite(gripper):
        raise XboxControlError("Robot driver returned a non-finite sent action.")
    return cart.lerobot_deg_to_model_rad(arm_deg), gripper


def command_self_test(args: argparse.Namespace) -> None:
    from xbox_teleoperator import XboxActionMapper, apply_deadzone, normalize_signed_axis

    if normalize_signed_axis(-32768) != -1.0 or normalize_signed_axis(32767) != 1.0:
        raise XboxControlError("Signed-axis normalization self-test failed.")
    if apply_deadzone(0.05, 0.10) != 0.0:
        raise XboxControlError("Deadzone self-test failed.")
    mapper = XboxActionMapper(deadzone=0.10, trigger_threshold=0.15)
    action = mapper.map_action(
        XboxState(left_x=1.0, left_y=-1.0, right_x=-0.5, right_y=0.5, right_trigger=1.0)
    )
    expected = {
        "delta_x": -1.0,
        "delta_y": 1.0,
        "delta_z": -(4.0 / 9.0),
        "delta_wrist_roll": 4.0 / 9.0,
        "gripper": 2,
    }
    if (
        not math.isclose(float(action["delta_x"]), expected["delta_x"])
        or not math.isclose(float(action["delta_y"]), expected["delta_y"])
        or not math.isclose(float(action["delta_z"]), expected["delta_z"])
        or not math.isclose(
            float(action["delta_wrist_roll"]), expected["delta_wrist_roll"]
        )
        or int(action["gripper"]) != expected["gripper"]
    ):
        raise XboxControlError(f"Xbox action mapping self-test failed: {action}")
    if mapper.map_events(XboxState(rb=True))["is_intervention"] is not True:
        raise XboxControlError("RB intervention mapping self-test failed.")
    print("Xbox mapping self-test passed.")

    cart = load_cartesian_module(args.project_root)
    kinematics = cart.SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
    q_reference = np.deg2rad(np.array([10.0, -25.0, 35.0, 15.0, -5.0]))
    pose = kinematics.fk(q_reference)
    result = kinematics.ik(pose, q_reference + np.deg2rad(2.0), orientation_weight=0.05)
    if not result.success:
        raise XboxControlError(f"Offline IK self-test failed: {result.message}")
    print(
        "SO-101 FK/IK self-test passed: "
        f"position_error={result.position_error_m * 1000.0:.3f} mm."
    )
    folded_q = np.deg2rad(np.array([0.0, 96.0, -108.0, 90.0, 0.0]))
    first_target, first_joint = sequential_unfold_target(
        cart, folded_q, command=1.0, dt=0.1, speed_deg_s=3.0
    )
    if first_joint != "wrist_flex" or first_target[3] >= folded_q[3]:
        raise XboxControlError("Sequential-unfold self-test failed for wrist_flex.")
    if not np.array_equal(first_target[[0, 1, 2, 4]], folded_q[[0, 1, 2, 4]]):
        raise XboxControlError("Sequential unfold moved more than one joint at once.")
    accumulated_target, accumulated_joint = sequential_unfold_target(
        cart,
        folded_q,
        command_q=first_target,
        command=1.0,
        dt=0.1,
        speed_deg_s=3.0,
    )
    if accumulated_joint != "wrist_flex" or accumulated_target[3] >= first_target[3]:
        raise XboxControlError("Sequential-unfold target did not accumulate across cycles.")
    elbow_phase = folded_q.copy()
    elbow_phase[3] = 0.0
    elbow_target, elbow_joint = sequential_unfold_target(
        cart, elbow_phase, command=1.0, dt=0.1, speed_deg_s=3.0
    )
    if elbow_joint != "elbow_flex" or elbow_target[2] <= elbow_phase[2]:
        raise XboxControlError("Sequential-unfold self-test failed for elbow_flex.")
    shoulder_phase = elbow_phase.copy()
    shoulder_phase[2] = 0.0
    shoulder_target, shoulder_joint = sequential_unfold_target(
        cart, shoulder_phase, command=1.0, dt=0.1, speed_deg_s=3.0
    )
    if shoulder_joint != "shoulder_lift" or shoulder_target[1] >= shoulder_phase[1]:
        raise XboxControlError("Sequential-unfold self-test failed for shoulder_lift.")
    print("Sequential-unfold self-test passed: wrist, elbow, then shoulder move toward 0 deg.")

    round_trip_action = {
        key: float(value)
        for key, value in zip(
            cart.ARM_OBSERVATION_KEYS,
            cart.model_rad_to_lerobot_deg(q_reference),
            strict=True,
        )
    }
    round_trip_action[cart.GRIPPER_KEY] = 12.5
    sent_q, sent_gripper = state_from_sent_action(cart, round_trip_action)
    if not np.allclose(sent_q, q_reference) or not math.isclose(sent_gripper, 12.5):
        raise XboxControlError("Sent-action state synchronization self-test failed.")
    print("Sent-action state synchronization self-test passed.")
    print("No controller or robot connection was made.")


def command_input_test(args: argparse.Namespace) -> None:
    teleop = make_teleop(args)
    print("INPUT TEST ONLY: this command never connects to the SO-101.")
    teleop.connect()
    print(f"Connected controller: {teleop.name}")
    print("Move every stick/trigger and press buttons. Only changes are printed. Press MENU to exit.")
    period = 1.0 / args.hz
    last_line: str | None = None
    try:
        while True:
            state = teleop.update()
            action = teleop.get_action(update=False)
            events = teleop.get_teleop_events()
            current_line = state_line(state, action)
            if current_line != last_line:
                print(current_line, flush=True)
                last_line = current_line
            if events["quit"]:
                print("MENU pressed. Input test finished.")
                return
            time.sleep(period)
    except KeyboardInterrupt:
        print("Input test stopped.")
    finally:
        teleop.disconnect()


def gripper_limits_from_calibration(calibration_path: Path, margin_deg: float = 2.0) -> tuple[float, float]:
    import json

    calibration = json.loads(calibration_path.expanduser().resolve().read_text())
    motor = calibration["gripper"]
    half_range_deg = float(motor["range_max"] - motor["range_min"]) * 180.0 / 4095.0
    if half_range_deg <= margin_deg:
        raise XboxControlError("Gripper calibration range is smaller than its safety margin.")
    return -half_range_deg + margin_deg, half_range_deg - margin_deg


def configured_original_pose(
    cart: ModuleType,
    current_gripper: float,
) -> tuple[np.ndarray, float]:
    """Load the user-edited original joint pose, refusing incomplete values."""

    arm_joint_names = tuple(cart.ARM_JOINTS)
    missing = [
        name
        for name in arm_joint_names
        if ORIGINAL_POSE_LEROBOT_DEG.get(name) is None
    ]
    if missing:
        raise XboxControlError(
            "Original pose is not configured. Fill these LeRobot-degree values in "
            "original_pose_config.py: " + ", ".join(missing)
        )

    arm_deg = np.asarray(
        [float(ORIGINAL_POSE_LEROBOT_DEG[name]) for name in arm_joint_names],
        dtype=float,
    )
    if not np.all(np.isfinite(arm_deg)):
        raise XboxControlError("Original arm pose contains NaN or infinity.")

    configured_gripper = ORIGINAL_POSE_LEROBOT_DEG.get("gripper")
    gripper = current_gripper if configured_gripper is None else float(configured_gripper)
    if not math.isfinite(gripper):
        raise XboxControlError("Original gripper position contains NaN or infinity.")

    return cart.lerobot_deg_to_model_rad(arm_deg), gripper


def needs_sequential_unfold(
    cart: ModuleType,
    q_arm_rad: np.ndarray,
    tolerance_deg: float = UNFOLD_DONE_TOLERANCE_DEG,
) -> bool:
    q_arm = np.asarray(q_arm_rad, dtype=float)
    for name in UNFOLD_JOINT_ORDER:
        index = cart.ARM_JOINTS.index(name)
        if abs(math.degrees(q_arm[index]) - UNFOLD_TARGET_DEG) > tolerance_deg:
            return True
    return False


def sequential_unfold_target(
    cart: ModuleType,
    measured_q: np.ndarray,
    *,
    command_q: np.ndarray | None = None,
    command: float,
    dt: float,
    speed_deg_s: float,
) -> tuple[np.ndarray, str | None]:
    """Move exactly one of joints 4, 3, 2 toward 0 degrees."""

    measured_q = np.asarray(measured_q, dtype=float)
    target_q = measured_q.copy() if command_q is None else np.asarray(command_q, dtype=float).copy()
    max_step = math.radians(speed_deg_s) * float(np.clip(command, 0.0, 1.0)) * max(dt, 0.0)
    target_rad = math.radians(UNFOLD_TARGET_DEG)
    tolerance_rad = math.radians(UNFOLD_DONE_TOLERANCE_DEG)

    for name in UNFOLD_JOINT_ORDER:
        index = cart.ARM_JOINTS.index(name)
        measured_error = target_rad - measured_q[index]
        if abs(measured_error) <= tolerance_rad:
            continue
        command_error = target_rad - target_q[index]
        if max_step > 0.0:
            target_q[index] += float(np.clip(command_error, -max_step, max_step))
        return target_q, name

    return target_q, None


def print_controls() -> None:
    print("\nXbox controls:")
    print("  Startup unfold: hold RB + push left stick forward")
    print("    order: wrist_flex -> elbow_flex -> shoulder_lift, each toward 0 deg")
    print("  Default JOINT mode (hold RB):")
    print("    left stick X/Y            base rotation / shoulder")
    print("    right stick Y/X           elbow / wrist rotation")
    print("    Y / X                     wrist flex + / -")
    print("  Click LEFT STICK with RB released to toggle JOINT / XYZ mode")
    print("  XYZ mode: left stick X/Y, right stick Y=Z, right stick X=wrist rotation")
    print("  Hold RB + RT / LT          Open / close gripper")
    print("  LB (with RB released)      Return once to configured original pose")
    print("  B                          Latch software E-stop and hold")
    print("  A (with RB released)       Reset software E-stop")
    print("  VIEW                       Print measured FK pose")
    print("  MENU                       Exit; prompts before torque release")


def command_auto_unfold_test(args: argparse.Namespace) -> None:
    """Move only the first unfinished unfold joint by a small amount and keep holding."""

    if not args.execute:
        raise XboxControlError("Auto motion is disabled because --execute was not provided.")

    cart = load_cartesian_module(args.project_root)
    driver = cart.SO101Driver(args.port, args.robot_id, args.max_relative_target_deg)
    torque_holding = False

    try:
        print(f"Connecting to SO-101 at {args.port} as '{args.robot_id}'...")
        driver.connect()
        observation = driver.observation()
        measured_q = driver.arm_q_model_rad(observation)
        hold_q = measured_q.copy()
        hold_gripper = float(observation[cart.GRIPPER_KEY])
        driver.send_arm_q_model_rad(hold_q, hold_gripper)
        torque_holding = True
        time.sleep(0.5)

        observation = driver.observation()
        measured_q = driver.arm_q_model_rad(observation)
        hold_q = measured_q.copy()
        hold_gripper = float(observation[cart.GRIPPER_KEY])
        driver.send_arm_q_model_rad(hold_q, hold_gripper)
        start_q = measured_q.copy()
        print("Settled pose captured; motor torque is holding the arm.")
        cart.print_joint_state(start_q, "Auto-test starting joint state")

        _preview_q, active_joint = sequential_unfold_target(
            cart,
            measured_q,
            command=1.0,
            dt=0.0,
            speed_deg_s=args.test_speed_deg_s,
        )
        if active_joint is None:
            print("All sequential-unfold joints are already within 5 deg of 0; no motion sent.")
        else:
            print(
                f"AUTO TEST: moving only {active_joint} toward 0 deg by at most "
                f"{args.test_motion_deg:.2f} deg."
            )
            period = 1.0 / args.hz
            deadline = time.monotonic() + args.test_motion_deg / args.test_speed_deg_s
            last_time = time.monotonic()
            while time.monotonic() < deadline:
                loop_start = time.monotonic()
                observation = driver.observation()
                measured_q = driver.arm_q_model_rad(observation)
                hold_gripper = float(observation[cart.GRIPPER_KEY])
                now = time.monotonic()
                dt = min(max(now - last_time, 0.0), period * 2.0)
                last_time = now
                hold_q, commanded_joint = sequential_unfold_target(
                    cart,
                    measured_q,
                    command_q=hold_q,
                    command=1.0,
                    dt=dt,
                    speed_deg_s=args.test_speed_deg_s,
                )
                if commanded_joint != active_joint:
                    break
                sent_action = driver.send_arm_q_model_rad(hold_q, hold_gripper)
                sent_deg = np.asarray(
                    [sent_action[key] for key in cart.ARM_OBSERVATION_KEYS], dtype=float
                )
                print(
                    compact_joint_line(cart, measured_q, hold_q)
                    + f" sent_{active_joint}={sent_deg[cart.ARM_JOINTS.index(active_joint)]:+.1f}",
                    flush=True,
                )
                elapsed = time.monotonic() - loop_start
                time.sleep(max(period - elapsed, 0.0))

        observation = driver.observation()
        measured_q = driver.arm_q_model_rad(observation)
        hold_q = measured_q.copy()
        hold_gripper = float(observation[cart.GRIPPER_KEY])
        driver.send_arm_q_model_rad(hold_q, hold_gripper)
        cart.print_joint_state(measured_q, "Auto-test final measured joint state")
        delta_deg = (
            cart.model_rad_to_lerobot_deg(measured_q)
            - cart.model_rad_to_lerobot_deg(start_q)
        )
        print(
            "Measured change deg: "
            + ", ".join(
                f"{name}={value:+.2f}" for name, value in zip(cart.ARM_JOINTS, delta_deg, strict=True)
            )
        )
        print("AUTO TEST COMPLETE. The arm remains torque-held.")
    finally:
        if driver.robot is not None and driver.robot.is_connected:
            if torque_holding:
                try:
                    observation = driver.observation()
                    measured_q = driver.arm_q_model_rad(observation)
                    driver.send_arm_q_model_rad(measured_q, float(observation[cart.GRIPPER_KEY]))
                except Exception as exc:
                    print(f"WARNING: could not refresh final hold command: {exc}")
                cart.wait_before_releasing_torque()
            driver.disconnect()


def command_control(args: argparse.Namespace) -> None:
    if not args.execute:
        raise XboxControlError(
            "Real control is disabled because --execute was not provided. "
            "Run 'input-test' first, then add --execute only after verifying every axis and button."
        )

    cart = load_cartesian_module(args.project_root)
    kinematics = cart.SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
    gripper_low, gripper_high = gripper_limits_from_calibration(args.calibration)
    teleop = make_teleop(args)
    driver = None
    torque_holding = False
    last_rb = False
    last_lb = False
    last_b = False
    last_a = False
    last_view = False
    last_left_stick = False
    estop_latched = False
    unfold_mode = False
    unfold_wait_for_rb_release = False
    active_unfold_joint: str | None = None
    control_mode = "JOINT"
    original_plan: list[np.ndarray] = []
    original_plan_index = 0
    original_gripper = 0.0

    try:
        print("Connecting to Xbox controller before touching the robot...")
        teleop.connect()
        print(f"Connected controller: {teleop.name}")
        state = teleop.update()
        if not controller_neutral(state):
            raise XboxControlError(
                "Controller is not neutral. Release RB, both triggers, and both sticks, then restart."
            )

        print(f"Connecting to SO-101 at {args.port} as '{args.robot_id}'...")
        driver = cart.SO101Driver(args.port, args.robot_id, args.max_relative_target_deg)
        driver.connect()
    except Exception:
        teleop.disconnect()
        if driver is not None and driver.robot is not None and driver.robot.is_connected:
            driver.disconnect()
        raise

    try:
        observation = driver.observation()
        measured_q = driver.arm_q_model_rad(observation)
        hold_q = measured_q.copy()
        hold_gripper = float(observation[cart.GRIPPER_KEY])
        # Capture and command the measured pose immediately so position control
        # supports the arm before any joystick motion is accepted.
        driver.send_arm_q_model_rad(hold_q, hold_gripper)
        torque_holding = True
        time.sleep(0.5)
        # Re-latch once after torque settles, avoiding a stale startup target.
        observation = driver.observation()
        measured_q = driver.arm_q_model_rad(observation)
        hold_q = measured_q.copy()
        hold_gripper = float(observation[cart.GRIPPER_KEY])
        driver.send_arm_q_model_rad(hold_q, hold_gripper)
        unfold_mode = needs_sequential_unfold(cart, hold_q)
        print("\nCURRENT POSE CAPTURED: motor position control is holding the settled pose.")
        print("Keep one hand near the power switch. Release the arm slowly only if stable.")
        cart.print_joint_state(hold_q, "Initial measured joint state")
        cart.print_pose("Initial FK pose", kinematics.fk(hold_q))
        if unfold_mode:
            print("SEQUENTIAL UNFOLD MODE: custom startup limit checks are disabled.")
            print("Hold RB and push the left stick forward; release either one to hold.")

        print_controls()
        print("\nMOTION ARMED. Movement occurs only while RB is held.")
        for remaining in (3, 2, 1):
            print(f"  {remaining}")
            time.sleep(1.0)

        period = 1.0 / args.hz
        session_start = time.monotonic()
        last_time = time.monotonic()
        last_status_time = 0.0

        while True:
            loop_start = time.monotonic()
            if (
                args.session_timeout_s > 0.0
                and loop_start - session_start >= args.session_timeout_s
            ):
                print(f"\nSession timeout reached ({args.session_timeout_s:.0f}s). Exiting control loop.")
                break
            state = teleop.update()
            action = teleop.get_action(update=False)
            events = teleop.get_teleop_events()
            now = time.monotonic()
            dt = min(max(now - last_time, 0.0), 0.10)
            last_time = now

            if events["quit"]:
                print("\nMENU pressed. Exiting control loop.")
                break

            observation = driver.observation()
            measured_q = driver.arm_q_model_rad(observation)
            measured_gripper = float(observation[cart.GRIPPER_KEY])

            if events["emergency_stop"] and not last_b:
                estop_latched = True
                unfold_mode = False
                unfold_wait_for_rb_release = False
                active_unfold_joint = None
                original_plan = []
                original_plan_index = 0
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                print("\nSOFTWARE E-STOP LATCHED: holding the measured pose.")

            if (
                events["reset_emergency_stop"]
                and not last_a
                and not state.rb
                and not state.b
            ):
                estop_latched = False
                unfold_mode = needs_sequential_unfold(cart, measured_q)
                unfold_wait_for_rb_release = False
                active_unfold_joint = None
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                print("\nSoftware E-stop reset.")

            if events["print_status"] and not last_view:
                cart.print_joint_state(measured_q, "\nMeasured joint state")
                cart.print_pose("Measured FK pose", kinematics.fk(measured_q))

            if (
                state.left_stick
                and not last_left_stick
                and not state.rb
                and not unfold_mode
                and not unfold_wait_for_rb_release
                and not estop_latched
            ):
                control_mode = "XYZ" if control_mode == "JOINT" else "JOINT"
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                print(f"\nControl mode changed to {control_mode}.")

            if events["return_to_original"] and not last_lb:
                if state.rb:
                    print("\nOriginal-pose request ignored: release RB before pressing LB.")
                elif estop_latched:
                    print("\nOriginal-pose request ignored: reset the software E-stop first.")
                elif unfold_mode or unfold_wait_for_rb_release:
                    print("\nOriginal-pose request ignored: finish sequential unfold first.")
                elif original_plan:
                    print("\nOriginal-pose return is already running.")
                else:
                    try:
                        original_q, original_gripper = configured_original_pose(
                            cart,
                            measured_gripper,
                        )
                        original_plan = kinematics.plan_joint_line(
                            measured_q,
                            original_q,
                            joint_step_rad=math.radians(args.original_joint_step_deg),
                        )
                        original_plan_index = 0
                        print(
                            "\nLB pressed: returning to the configured original pose "
                            f"in {len(original_plan)} small joint-space steps."
                        )
                        print("Press B to stop and hold, or press RB to cancel and take manual control.")
                    except Exception as exc:
                        original_plan = []
                        original_plan_index = 0
                        print(f"\nOriginal-pose return was not started: {exc}")

            if original_plan and state.rb:
                original_plan = []
                original_plan_index = 0
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                print("\nRB pressed: original-pose return cancelled; manual control enabled.")

            if unfold_wait_for_rb_release and not state.rb:
                unfold_wait_for_rb_release = False
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                print("\nRB released: normal Cartesian control is now armed.")

            movement_enabled = state.rb and not estop_latched and not unfold_mode and not unfold_wait_for_rb_release
            if state.rb and not last_rb:
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                if unfold_mode:
                    print("\nRB enabled for sequential unfold; push the left stick forward.")
                elif unfold_wait_for_rb_release:
                    print("\nRelease RB once before normal Cartesian control.")
                elif not estop_latched:
                    print("\nRB enabled.")
            if not state.rb and last_rb:
                hold_q = measured_q.copy()
                hold_gripper = measured_gripper
                print("\nRB released: holding current pose.")

            if unfold_mode and not estop_latched:
                unfold_command = max(0.0, float(action["delta_y"])) if state.rb else 0.0
                previous_unfold_joint = active_unfold_joint
                unfold_q, active_unfold_joint = sequential_unfold_target(
                    cart,
                    measured_q,
                    command_q=hold_q,
                    command=unfold_command,
                    dt=dt,
                    speed_deg_s=args.unfold_speed_deg_s,
                )
                if active_unfold_joint is not None and active_unfold_joint != previous_unfold_joint:
                    motor_id = cart.ARM_JOINTS.index(active_unfold_joint) + 1
                    print(f"\nUNFOLD PHASE: motor {motor_id} {active_unfold_joint} -> 0 deg")
                if active_unfold_joint is None:
                    unfold_mode = False
                    unfold_wait_for_rb_release = state.rb
                    hold_q = measured_q.copy()
                    hold_gripper = measured_gripper
                    print("\nSEQUENTIAL UNFOLD COMPLETE: joints 4, 3, and 2 are near 0 deg.")
                    if state.rb:
                        print("Release RB once before using normal Cartesian control.")
                elif unfold_command > 0.0:
                    hold_q = unfold_q
                    hold_gripper = measured_gripper
            elif original_plan and not estop_latched:
                hold_q = original_plan[original_plan_index].copy()
                hold_gripper = original_gripper
                original_plan_index += 1
                if original_plan_index >= len(original_plan):
                    original_plan = []
                    original_plan_index = 0
                    print("\nOriginal-pose commands complete: holding the configured target.")
            elif movement_enabled and control_mode == "JOINT":
                joint_commands = np.asarray(
                    [
                        float(action["delta_x"]),
                        float(action["delta_y"]),
                        float(action["delta_z"]),
                        float(state.y) - float(state.x),
                        float(action["delta_wrist_roll"]),
                    ],
                    dtype=float,
                )
                hold_q = np.clip(
                    hold_q
                    + np.deg2rad(args.joint_speed_deg_s * dt * joint_commands),
                    kinematics.lower,
                    kinematics.upper,
                )

                gripper_direction = 0.0
                if int(action["gripper"]) == int(GripperAction.OPEN):
                    gripper_direction = 1.0
                elif int(action["gripper"]) == int(GripperAction.CLOSE):
                    gripper_direction = -1.0
                hold_gripper = float(
                    np.clip(
                        hold_gripper + gripper_direction * args.gripper_speed_deg_s * dt,
                        gripper_low,
                        gripper_high,
                    )
                )
            elif movement_enabled:
                delta = np.array(
                    [float(action["delta_x"]), float(action["delta_y"]), float(action["delta_z"])],
                    dtype=float,
                )
                delta_m = delta * (args.linear_speed_mm_s / 1000.0) * dt
                delta_norm = float(np.linalg.norm(delta_m))
                max_step_m = args.max_cartesian_step_mm / 1000.0
                if delta_norm > max_step_m:
                    delta_m *= max_step_m / delta_norm

                current_pose = kinematics.fk(measured_q)
                target_pose = cart.Pose(
                    xyz_m=current_pose.xyz_m + delta_m,
                    rotation=current_pose.rotation.copy(),
                )

                gripper_direction = 0.0
                if int(action["gripper"]) == int(GripperAction.OPEN):
                    gripper_direction = 1.0
                elif int(action["gripper"]) == int(GripperAction.CLOSE):
                    gripper_direction = -1.0
                gripper_target = float(
                    np.clip(
                        measured_gripper + gripper_direction * args.gripper_speed_deg_s * dt,
                        gripper_low,
                        gripper_high,
                    )
                )

                if delta_norm > 0.0:
                    try:
                        plan = kinematics.plan_cartesian_line(
                            measured_q,
                            target_pose,
                            orientation_weight=args.orientation_weight,
                            cartesian_step_m=max_step_m,
                        )
                        hold_q = plan[-1].copy()
                    except Exception as exc:
                        hold_q = measured_q.copy()
                        hold_gripper = measured_gripper
                        print(f"\nCartesian plan rejected; holding current pose: {exc}")
                hold_gripper = gripper_target

                wrist_roll_index = cart.ARM_JOINTS.index("wrist_roll")
                hold_q[wrist_roll_index] = float(
                    np.clip(
                        hold_q[wrist_roll_index]
                        + math.radians(args.joint_speed_deg_s)
                        * dt
                        * float(action["delta_wrist_roll"]),
                        kinematics.lower[wrist_roll_index],
                        kinematics.upper[wrist_roll_index],
                    )
                )

            sent_action = driver.send_arm_q_model_rad(hold_q, hold_gripper)
            # LeRobot can clamp a target relative to the measured position.
            # Continue from the value actually sent so the internal command
            # cannot run ahead of the robot and flood the console with warnings.
            hold_q, hold_gripper = state_from_sent_action(cart, sent_action)

            if now - last_status_time >= 0.5:
                if estop_latched:
                    mode = "E-STOP"
                elif unfold_mode:
                    mode = f"UNFOLD-{active_unfold_joint or 'WAIT'}"
                elif unfold_wait_for_rb_release:
                    mode = "RELEASE-RB"
                elif original_plan:
                    mode = "ORIGIN"
                else:
                    mode = control_mode if movement_enabled else f"HOLD-{control_mode}"
                pose = kinematics.fk(measured_q)
                print(
                    "\r"
                    f"mode={mode:<13s} xyz_mm="
                    f"[{pose.xyz_m[0] * 1000:7.1f},"
                    f"{pose.xyz_m[1] * 1000:7.1f},"
                    f"{pose.xyz_m[2] * 1000:7.1f}] "
                    f"{compact_joint_line(cart, measured_q, hold_q)} "
                    f"{state_line(state, action)}".ljust(220),
                    end="",
                    flush=True,
                )
                last_status_time = now

            last_rb = state.rb
            last_lb = state.lb
            last_b = state.b
            last_a = state.a
            last_view = state.view
            last_left_stick = state.left_stick
            elapsed = time.monotonic() - loop_start
            time.sleep(max(period - elapsed, 0.0))

    except KeyboardInterrupt:
        print("\nStopped by keyboard.")
    finally:
        teleop.disconnect()
        if driver is not None and driver.robot is not None and driver.robot.is_connected:
            if torque_holding:
                try:
                    observation = driver.observation()
                    measured_q = driver.arm_q_model_rad(observation)
                    driver.send_arm_q_model_rad(
                        measured_q,
                        float(observation[cart.GRIPPER_KEY]),
                    )
                except Exception as exc:
                    print(f"\nWARNING: could not refresh final hold command: {exc}")
                cart.wait_before_releasing_torque()
            driver.disconnect()


def add_controller_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--deadzone", type=float, default=0.10)
    parser.add_argument("--trigger-threshold", type=float, default=0.15)
    parser.add_argument(
        "--allow-non-xbox",
        action="store_true",
        help="Allow any SDL GameController name. Intended only for adapter debugging.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Xbox Series X|S controller adapter for SO-101.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "simulation" / "so101_new_calib.urdf",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "config" / "my_follower.json",
    )
    parser.add_argument("--joint-margin-deg", type=float, default=3.0)

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="Offline adapter and FK/IK tests.")

    input_test = commands.add_parser("input-test", help="Display controller input; never connects to robot.")
    add_controller_arguments(input_test)
    input_test.add_argument("--hz", type=float, default=20.0)

    auto_test = commands.add_parser(
        "auto-test",
        help="Move the first unfinished unfold joint by a small amount; no controller required.",
    )
    auto_test.add_argument("--execute", action="store_true")
    auto_test.add_argument("--port", default="/dev/cu.usbmodem5B3E0888091")
    auto_test.add_argument("--robot-id", default="my_follower")
    auto_test.add_argument("--hz", type=float, default=10.0)
    auto_test.add_argument("--max-relative-target-deg", type=float, default=5.0)
    auto_test.add_argument("--test-speed-deg-s", type=float, default=3.0)
    auto_test.add_argument("--test-motion-deg", type=float, default=15.0)

    control = commands.add_parser("control", help="Control the real SO-101; requires --execute.")
    add_controller_arguments(control)
    control.add_argument("--execute", action="store_true")
    control.add_argument("--port", default="/dev/cu.usbmodem5B3E0888091")
    control.add_argument("--robot-id", default="my_follower")
    control.add_argument("--hz", type=float, default=15.0)
    control.add_argument("--linear-speed-mm-s", type=float, default=20.0)
    control.add_argument("--max-cartesian-step-mm", type=float, default=2.0)
    control.add_argument("--gripper-speed-deg-s", type=float, default=25.0)
    control.add_argument(
        "--joint-speed-deg-s",
        type=float,
        default=30.0,
        help="Direct joint and wrist-roll jog speed.",
    )
    control.add_argument("--orientation-weight", type=float, default=0.05)
    control.add_argument("--max-relative-target-deg", type=float, default=5.0)
    control.add_argument(
        "--unfold-speed-deg-s",
        type=float,
        default=8.0,
        help="Sequential startup-unfold speed for joints 4, 3, and 2.",
    )
    control.add_argument(
        "--original-joint-step-deg",
        type=float,
        default=0.75,
        help="Maximum commanded joint-space step while returning to the original pose.",
    )
    control.add_argument(
        "--session-timeout-s",
        type=float,
        default=0.0,
        help="Automatically stop after this many seconds. Zero disables the timeout.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.joint_margin_deg <= 10.0:
        raise XboxControlError("--joint-margin-deg must be within [0, 10].")
    if args.command in {"input-test", "control"}:
        if not 0.0 <= args.deadzone < 0.5:
            raise XboxControlError("--deadzone must be within [0, 0.5).")
        if not 0.0 <= args.trigger_threshold <= 1.0:
            raise XboxControlError("--trigger-threshold must be within [0, 1].")
        if not 1.0 <= args.hz <= 30.0:
            raise XboxControlError("--hz must be within [1, 30].")
    if args.command == "auto-test":
        if not 1.0 <= args.hz <= 20.0:
            raise XboxControlError("--hz must be within [1, 20] for auto-test.")
        if not 0.25 <= args.max_relative_target_deg <= 10.0:
            raise XboxControlError("--max-relative-target-deg must be within [0.25, 10].")
        if not 0.25 <= args.test_speed_deg_s <= 5.0:
            raise XboxControlError("--test-speed-deg-s must be within [0.25, 5].")
        if not 0.25 <= args.test_motion_deg <= 20.0:
            raise XboxControlError("--test-motion-deg must be within [0.25, 20].")
    if args.command == "control":
        if not 1.0 <= args.linear_speed_mm_s <= 50.0:
            raise XboxControlError("--linear-speed-mm-s must be within [1, 50].")
        if not 0.5 <= args.max_cartesian_step_mm <= 3.0:
            raise XboxControlError("--max-cartesian-step-mm must be within [0.5, 3].")
        if not 1.0 <= args.gripper_speed_deg_s <= 50.0:
            raise XboxControlError("--gripper-speed-deg-s must be within [1, 50].")
        if not 1.0 <= args.joint_speed_deg_s <= 60.0:
            raise XboxControlError("--joint-speed-deg-s must be within [1, 60].")
        if not 0.0 <= args.orientation_weight <= 0.3:
            raise XboxControlError("--orientation-weight must be within [0, 0.3].")
        if not 0.25 <= args.max_relative_target_deg <= 10.0:
            raise XboxControlError("--max-relative-target-deg must be within [0.25, 10].")
        if not 0.5 <= args.unfold_speed_deg_s <= 15.0:
            raise XboxControlError("--unfold-speed-deg-s must be within [0.5, 15].")
        if not 0.25 <= args.original_joint_step_deg <= 1.5:
            raise XboxControlError("--original-joint-step-deg must be within [0.25, 1.5].")
        if args.session_timeout_s != 0.0 and not 10.0 <= args.session_timeout_s <= 900.0:
            raise XboxControlError("--session-timeout-s must be 0 or within [10, 900].")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        if args.command == "self-test":
            command_self_test(args)
        elif args.command == "input-test":
            command_input_test(args)
        elif args.command == "auto-test":
            command_auto_unfold_test(args)
        elif args.command == "control":
            command_control(args)
        else:
            parser.error(f"Unknown command: {args.command}")
        return 0
    except (XboxControlError, XboxInputError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
