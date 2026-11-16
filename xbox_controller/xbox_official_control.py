#!/usr/bin/env python3
"""LeRobot HIL-style Xbox control for an SO-101.

Pipeline: Xbox XYZ deltas -> measured EE reference -> bounds/IK -> joint targets.
The folded HOME pose is the storage/start pose. A separately measured NEUTRAL
working pose is used for a deadman-controlled, all-joint smooth deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from xbox_teleoperator import GripperAction, XboxInputError, XboxLeRobotTeleop, XboxState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDED_POSE_PATH = Path(__file__).resolve().parent / "folded_pose.json"
NEUTRAL_POSE_PATH = Path(__file__).resolve().parent / "neutral_pose.json"
ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class ControlError(RuntimeError):
    pass


def cart_module():
    sys.path.insert(0, str(PROJECT_ROOT))
    import so101_cartesian_control as cart

    return cart


def neutral(state: XboxState, tolerance: float = 0.20) -> bool:
    return (
        max(abs(state.left_x), abs(state.left_y), abs(state.right_x), abs(state.right_y))
        <= tolerance
        and state.left_trigger <= tolerance
        and state.right_trigger <= tolerance
        and not state.rb
    )


def sent_state(cart, action: dict[str, float]) -> tuple[np.ndarray, float]:
    values = np.asarray([float(action[key]) for key in cart.ARM_OBSERVATION_KEYS])
    gripper = float(action[cart.GRIPPER_KEY])
    if not np.all(np.isfinite(values)) or not math.isfinite(gripper):
        raise ControlError("Driver returned a non-finite action.")
    return cart.lerobot_deg_to_model_rad(values), gripper


def send_with_cached_safety_limit(
    driver,
    cart,
    q_arm_rad: np.ndarray,
    gripper: float,
    observation: dict[str, float],
    max_relative_target: float,
) -> dict[str, float]:
    """Clamp against this cycle's observation, then write without another bus read."""

    desired = np.append(cart.model_rad_to_lerobot_deg(q_arm_rad), float(gripper))
    keys = (*cart.ARM_OBSERVATION_KEYS, cart.GRIPPER_KEY)
    present = np.asarray([float(observation[key]) for key in keys], dtype=float)
    safe = np.clip(desired, present - max_relative_target, present + max_relative_target)
    safe_q = cart.lerobot_deg_to_model_rad(safe[:-1])
    return driver.send_arm_q_model_rad(safe_q, float(safe[-1]))


def load_operating_pose(path: Path, cart) -> tuple[np.ndarray, float | None] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    joints = data.get("joints_lerobot_deg", {})
    if any(joints.get(name) is None for name in ARM_NAMES):
        return None
    q = cart.lerobot_deg_to_model_rad([float(joints[name]) for name in ARM_NAMES])
    gripper = joints.get("gripper")
    return q, None if gripper is None else float(gripper)


def save_operating_pose(path: Path, cart, q: np.ndarray, gripper: float) -> None:
    degrees = cart.model_rad_to_lerobot_deg(q)
    data = {
        "format": "lerobot_degrees",
        "description": "Measured, user-approved Xbox teleoperation neutral pose",
        "joints_lerobot_deg": {
            **{name: round(float(value), 6) for name, value in zip(ARM_NAMES, degrees, strict=True)},
            "gripper": round(float(gripper), 6),
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def reset_plan(kinematics, start: np.ndarray, goal: np.ndarray, speed: float, hz: float):
    """Synchronized quintic S-curve with zero velocity/acceleration at both ends."""

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != (5,) or goal.shape != (5,) or not np.all(np.isfinite([start, goal])):
        raise ControlError("HOME/NEUTRAL planning requires two finite five-joint poses.")
    tolerance = math.radians(1.0)
    if np.any(goal < kinematics.hard_lower - tolerance) or np.any(
        goal > kinematics.hard_upper + tolerance
    ):
        raise ControlError("HOME/NEUTRAL target exceeds a calibrated hard joint limit.")
    max_distance = float(np.max(np.abs(goal - start)))
    if max_distance == 0.0:
        return []
    # Quintic smoothstep's peak normalized velocity is 1.875. Scale duration so
    # `speed` remains a true peak joint-speed limit rather than an average.
    duration = 1.875 * max_distance / math.radians(speed)
    steps = max(1, int(math.ceil(duration * hz)))
    plan = []
    for index in range(1, steps + 1):
        t = index / steps
        blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        plan.append(start + blend * (goal - start))
    return plan


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default="/dev/cu.usbmodem5B3E0888091")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--urdf", type=Path, default=PROJECT_ROOT / "simulation/so101_new_calib.urdf")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "config/my_follower.json")
    parser.add_argument("--neutral-pose", type=Path, default=NEUTRAL_POSE_PATH)
    parser.add_argument("--joint-margin-deg", type=float, default=3.0)
    parser.add_argument("--max-relative-target-deg", type=float, default=8.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot HIL-style Xbox control for SO-101")
    commands = parser.add_subparsers(dest="command", required=True)
    test = commands.add_parser("self-test")
    common_parser(test)
    capture = commands.add_parser("capture-neutral")
    common_parser(capture)
    capture.add_argument("--execute", action="store_true")
    control = commands.add_parser("control")
    common_parser(control)
    control.add_argument("--execute", action="store_true")
    control.add_argument("--controller-index", type=int, default=0)
    control.add_argument("--allow-non-xbox", action="store_true")
    control.add_argument("--deadzone", type=float, default=0.10)
    control.add_argument("--trigger-threshold", type=float, default=0.15)
    control.add_argument("--hz", type=float, default=20.0)
    control.add_argument("--linear-speed-mm-s", type=float, default=15.0)
    control.add_argument("--max-cartesian-step-mm", type=float, default=1.5)
    control.add_argument("--orientation-weight", type=float, default=0.05)
    control.add_argument("--wrist-roll-speed-deg-s", type=float, default=35.0)
    control.add_argument("--gripper-speed-deg-s", type=float, default=20.0)
    control.add_argument("--deploy-speed-deg-s", type=float, default=30.0)
    return parser


def validate(args: argparse.Namespace) -> None:
    if not 0.0 <= args.joint_margin_deg <= 10.0:
        raise ControlError("--joint-margin-deg must be within [0, 10].")
    if not 0.25 <= args.max_relative_target_deg <= 10.0:
        raise ControlError("--max-relative-target-deg must be within [0.25, 10].")
    if args.command == "control":
        checks = (
            (1 <= args.hz <= 30, "--hz must be within [1, 30]."),
            (1 <= args.linear_speed_mm_s <= 50, "--linear-speed-mm-s must be within [1, 50]."),
            (0.5 <= args.max_cartesian_step_mm <= 3, "--max-cartesian-step-mm must be within [0.5, 3]."),
            (1 <= args.wrist_roll_speed_deg_s <= 60, "--wrist-roll-speed-deg-s must be within [1, 60]."),
            (1 <= args.gripper_speed_deg_s <= 50, "--gripper-speed-deg-s must be within [1, 50]."),
            (1 <= args.deploy_speed_deg_s <= 30, "--deploy-speed-deg-s must be within [1, 30]."),
        )
        for ok, message in checks:
            if not ok:
                raise ControlError(message)


def command_self_test(args: argparse.Namespace) -> None:
    cart = cart_module()
    kin = cart.SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
    q = np.deg2rad([0.0, -30.0, 60.0, -30.0, 0.0])
    pose = kin.fk(q)
    result = kin.ik(pose, q + np.deg2rad(1.0), orientation_weight=0.05)
    if not result.success or result.position_error_m > 0.001:
        raise ControlError(f"FK/IK failed: {result.message}")
    plan = reset_plan(kin, q, q + np.deg2rad([5, -5, 5, -5, 5]), 10.0, 10.0)
    increments = np.diff(np.vstack([q, *plan]), axis=0)
    increment_sizes = np.max(np.abs(np.rad2deg(increments)), axis=1)
    if (
        not plan
        or not np.allclose(plan[-1], q + np.deg2rad([5, -5, 5, -5, 5]))
        or float(np.max(increment_sizes)) > 1.000001
        or increment_sizes[0] >= increment_sizes[len(increment_sizes) // 2]
        or increment_sizes[-1] >= increment_sizes[len(increment_sizes) // 2]
    ):
        raise ControlError("Smooth deployment plan failed.")
    fake = dict(zip(cart.ARM_OBSERVATION_KEYS, cart.model_rad_to_lerobot_deg(q), strict=True))
    fake[cart.GRIPPER_KEY] = 12.0
    q2, g2 = sent_state(cart, fake)
    if not np.allclose(q, q2) or not math.isclose(g2, 12.0):
        raise ControlError("Sent-action state synchronization failed.")

    class OfflineDriver:
        @staticmethod
        def send_arm_q_model_rad(q_arm_rad, gripper_value):
            result = dict(
                zip(
                    cart.ARM_OBSERVATION_KEYS,
                    cart.model_rad_to_lerobot_deg(q_arm_rad),
                    strict=True,
                )
            )
            result[cart.GRIPPER_KEY] = float(gripper_value)
            return result

    cached_obs = dict(fake)
    limited = send_with_cached_safety_limit(
        OfflineDriver(), cart, q + math.radians(20.0), 32.0, cached_obs, 8.0
    )
    limited_values = np.asarray([limited[key] for key in (*cart.ARM_OBSERVATION_KEYS, cart.GRIPPER_KEY)])
    present_values = np.asarray([cached_obs[key] for key in (*cart.ARM_OBSERVATION_KEYS, cart.GRIPPER_KEY)])
    if np.max(np.abs(limited_values - present_values)) > 8.000001:
        raise ControlError("Cached-observation safety limiting failed.")

    home_pose = load_operating_pose(FOLDED_POSE_PATH, cart)
    if home_pose is None:
        raise ControlError(f"HOME pose is missing: {FOLDED_POSE_PATH}")
    neutral_pose = load_operating_pose(args.neutral_pose, cart)
    home_neutral_plan = []
    max_radius = 0.0
    if neutral_pose is not None:
        home_neutral_plan = reset_plan(kin, home_pose[0], neutral_pose[0], 15.0, 10.0)
        for sample_q in [home_pose[0], *home_neutral_plan]:
            if np.any(sample_q < kin.hard_lower - math.radians(1.0)) or np.any(
                sample_q > kin.hard_upper + math.radians(1.0)
            ):
                raise ControlError("HOME-to-NEUTRAL trajectory exceeds a calibrated hard limit.")
            max_radius = max(max_radius, float(np.linalg.norm(kin.fk(sample_q).xyz_m)))
        if max_radius > 0.50:
            raise ControlError("HOME-to-NEUTRAL trajectory exceeds the 0.50 m radius envelope.")
    print("Official-style controller self-test passed.")
    print(f"FK/IK error={result.position_error_m * 1000:.3f} mm, deployment_steps={len(plan)}")
    if neutral_pose is None:
        print(f"HOME configured; NEUTRAL not captured ({args.neutral_pose}).")
    else:
        print(
            f"HOME-to-NEUTRAL trajectory: {len(home_neutral_plan)} steps, "
            f"max_radius={max_radius * 1000:.1f} mm"
        )
    print("No controller or robot connection was made.")


def driver_for(args: argparse.Namespace, cart):
    # Safety limiting is performed from the observation already read by this
    # controller. Leaving LeRobot's limiter disabled avoids its second bus read.
    driver = cart.SO101Driver(args.port, args.robot_id, None)
    driver.connect()
    return driver


def command_capture(args: argparse.Namespace) -> None:
    if not args.execute:
        raise ControlError("Capture requires --execute.")
    cart = cart_module()
    kin = cart.SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    try:
        # Connect only the bus and explicitly keep torque disabled. Calling the
        # full robot.connect() here would configure and re-enable position hold
        # before the user has finished placing the arm.
        robot.bus.connect()
        robot.bus.disable_torque()
        print("Motor torque is OFF. Support the arm and place it in the desired NEUTRAL working pose.")
        input("Press ENTER when the pose is stable...")
        positions = robot.bus.sync_read("Present_Position")
        obs = {f"{name}.pos": float(value) for name, value in positions.items()}
        from so101_events import report_position

        report_position(obs)
        q = cart.lerobot_deg_to_model_rad([obs[key] for key in cart.ARM_OBSERVATION_KEYS])
        violations = kin.joint_limit_violations(q)
        if violations:
            raise ControlError("Neutral pose outside limits: " + "; ".join(violations))
        cart.print_joint_state(q, "Measured NEUTRAL pose")
        cart.print_pose("Measured NEUTRAL FK", kin.fk(q))
        save_operating_pose(args.neutral_pose, cart, q, float(obs[cart.GRIPPER_KEY]))
        print(f"Saved: {args.neutral_pose}")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)


def print_controls(has_operating_pose: bool) -> None:
    print("\nControls:")
    print("  JOINT mode: RB + left X/Y   base / shoulder")
    print("  JOINT mode: RB + right Y/X  elbow / wrist roll")
    print("  JOINT mode: RB + Y / X      wrist flex +/-")
    print("  click LEFT STICK with RB released to toggle JOINT / XYZ")
    print("  XYZ mode: RB + left X/Y, right Y=Z, right X=wrist roll")
    print("  RB + RT / LT            open / close gripper")
    print("  RB + tap LB             toggle smoothly between HOME and NEUTRAL")
    print("  release RB              hold immediately / cancel deployment")
    print("  B                       software E-stop")
    print("  A with RB released      reset software E-stop")
    print("  VIEW                    print joints and FK")
    print("  MENU                    exit")
    if not has_operating_pose:
        print("  NOTE: deployment is disabled until a NEUTRAL pose is captured.")


def control_steps(
    args: argparse.Namespace,
    *,
    teleop_factory=XboxLeRobotTeleop,
    driver_factory=driver_for,
    observation_callback=None,
    action_callback=None,
):
    """Yield after each control cycle for either the CLI loop or a ROS timer."""
    if not args.execute:
        raise ControlError("Real control requires --execute.")
    cart = cart_module()
    kin = cart.SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
    home_pose = load_operating_pose(FOLDED_POSE_PATH, cart)
    if home_pose is None:
        raise ControlError(f"HOME pose is missing: {FOLDED_POSE_PATH}")
    operating_pose = load_operating_pose(args.neutral_pose, cart)
    # SO-101 gripper observations/actions use LeRobot's normalized 0..100 range,
    # unlike the five arm joints which use calibrated degrees.
    grip_limits = (2.0, 98.0)
    teleop = teleop_factory(
        controller_index=args.controller_index,
        deadzone=args.deadzone,
        trigger_threshold=args.trigger_threshold,
        require_xbox_name=not args.allow_non_xbox,
    )
    driver = None
    torque_holding = False
    print("Connecting to Xbox controller before touching the robot...")
    teleop.connect()
    try:
        state = teleop.update()
        if not neutral(state):
            raise ControlError("Release RB, triggers, and both sticks before startup.")
        print(f"Connected controller: {teleop.name}")
        driver = driver_factory(args, cart)
        obs = driver.observation()
        measured_q = driver.arm_q_model_rad(obs)
        measured_g = float(obs[cart.GRIPPER_KEY])
        hold_q, hold_g = measured_q.copy(), measured_g
        send_with_cached_safety_limit(
            driver, cart, hold_q, hold_g, obs, args.max_relative_target_deg
        )
        torque_holding = True
        time.sleep(0.5)
        obs = driver.observation()
        measured_q = driver.arm_q_model_rad(obs)
        measured_g = float(obs[cart.GRIPPER_KEY])
        hold_q, hold_g = measured_q.copy(), measured_g
        send_with_cached_safety_limit(
            driver, cart, hold_q, hold_g, obs, args.max_relative_target_deg
        )
        print("\nFolded/current pose captured and held. No automatic movement was sent.")
        cart.print_joint_state(measured_q, "Initial joints")
        cart.print_pose("Initial FK", kin.fk(measured_q))
        cart.print_joint_state(home_pose[0], "Configured HOME pose")
        if operating_pose is not None:
            cart.print_joint_state(operating_pose[0], "Configured NEUTRAL pose")
            cart.print_pose("Configured NEUTRAL FK", kin.fk(operating_pose[0]))
        else:
            print(f"Configured NEUTRAL pose: NOT CAPTURED ({args.neutral_pose})")
        print_controls(operating_pose is not None)

        period = 1.0 / args.hz
        last_time = time.monotonic()
        last_status = 0.0
        rate_started = last_time
        rate_cycles = 0
        actual_hz = 0.0
        last_rb = last_lb = last_b = last_a = last_view = last_left_stick = False
        estop = False
        control_mode = "JOINT"
        deploy: list[np.ndarray] = []
        deploy_index = 0
        deploy_destination = ""
        operating_g = measured_g
        at_home = np.max(np.abs(np.rad2deg(measured_q - home_pose[0]))) <= 10.0
        operating_reached = (
            operating_pose is not None
            and np.max(np.abs(np.rad2deg(measured_q - operating_pose[0]))) <= 10.0
        )
        rearm_required = False
        if not operating_reached:
            print("\nHOME/STARTUP LOCKED: only RB + LB deployment is accepted.")

        while True:
            started = time.monotonic()
            state = teleop.update()
            action = teleop.get_action(update=False)
            events = teleop.get_teleop_events()
            now = time.monotonic()
            dt = min(max(now - last_time, 0.0), 0.1)
            last_time = now
            if events["quit"]:
                print("\nMENU pressed. Exiting.")
                break
            obs = driver.observation()
            if observation_callback is not None:
                observation_callback(obs)
            measured_q = driver.arm_q_model_rad(obs)
            measured_g = float(obs[cart.GRIPPER_KEY])

            if events["emergency_stop"] and not last_b:
                estop, deploy, deploy_index = True, [], 0
                hold_q, hold_g = measured_q.copy(), measured_g
                print("\nSOFTWARE E-STOP LATCHED.")
            if events["reset_emergency_stop"] and not last_a and not state.rb and not state.b:
                estop = False
                hold_q, hold_g = measured_q.copy(), measured_g
                print("\nSoftware E-stop reset.")
            if events["print_status"] and not last_view:
                cart.print_joint_state(measured_q, "\nMeasured joints")
                cart.print_pose("Measured FK", kin.fk(measured_q))

            if (
                state.left_stick and not last_left_stick and not state.rb
                and operating_reached and not estop and not deploy
            ):
                control_mode = "XYZ" if control_mode == "JOINT" else "JOINT"
                hold_q, hold_g = measured_q.copy(), measured_g
                print(f"\nControl mode changed to {control_mode}.")

            if state.rb and state.lb and not last_lb and not estop:
                if operating_reached:
                    target_q, saved_g = home_pose
                    operating_g = measured_g if saved_g is None else saved_g
                    deploy_destination = "HOME"
                elif operating_pose is None:
                    print("\nDeployment disabled: capture a verified NEUTRAL pose first.")
                else:
                    operating_q, saved_g = operating_pose
                    operating_g = measured_g if saved_g is None else saved_g
                    target_q = operating_q
                    deploy_destination = "NEUTRAL"
                if deploy_destination:
                    deploy = reset_plan(kin, measured_q, target_q, args.deploy_speed_deg_s, args.hz)
                    deploy_index = 0
                    operating_reached = False
                    at_home = False
                    rearm_required = False
                    if not deploy:
                        operating_reached = deploy_destination == "NEUTRAL"
                        at_home = deploy_destination == "HOME"
                        rearm_required = True
                        print(f"\nAlready at {deploy_destination} pose. Release RB once.")
                    else:
                        print(
                            f"\nMOVE TO {deploy_destination}: {len(deploy)} synchronized steps. "
                            "Keep holding RB."
                        )

            if not state.rb and last_rb:
                if deploy:
                    print(f"\nRB released: move to {deploy_destination} cancelled and pose held.")
                deploy, deploy_index = [], 0
                deploy_destination = ""
                hold_q, hold_g = measured_q.copy(), measured_g
                rearm_required = False

            enabled = state.rb and not estop and not rearm_required
            if deploy and enabled:
                hold_q, hold_g = deploy[deploy_index].copy(), operating_g
                deploy_index += 1
                if deploy_index >= len(deploy):
                    deploy, deploy_index = [], 0
                    operating_reached = deploy_destination == "NEUTRAL"
                    at_home = deploy_destination == "HOME"
                    rearm_required = True
                    print(f"\n{deploy_destination} POSE REACHED. Release RB once.")
                    deploy_destination = ""
            elif enabled and operating_reached and not state.lb and control_mode == "JOINT":
                joint_commands = np.asarray(
                    [
                        # `delta_x` follows Cartesian coordinates and is the
                        # opposite of the direct shoulder-pan joystick sense.
                        -float(action["delta_x"]),
                        float(action["delta_y"]),
                        float(action["delta_z"]),
                        float(state.y) - float(state.x),
                        float(action["delta_wrist_roll"]),
                    ],
                    dtype=float,
                )
                hold_q = np.clip(
                    hold_q + np.deg2rad(25.0 * dt * joint_commands),
                    kin.lower,
                    kin.upper,
                )
                grip_direction = (
                    1.0 if action["gripper"] == GripperAction.OPEN
                    else -1.0 if action["gripper"] == GripperAction.CLOSE
                    else 0.0
                )
                hold_g = float(np.clip(
                    hold_g + grip_direction * args.gripper_speed_deg_s * dt,
                    *grip_limits,
                ))
                at_home = False
            elif enabled and operating_reached and not state.lb:
                delta = np.asarray([action["delta_x"], action["delta_y"], action["delta_z"]], float)
                delta_m = delta * args.linear_speed_mm_s / 1000.0 * dt
                norm = float(np.linalg.norm(delta_m))
                max_step = args.max_cartesian_step_mm / 1000.0
                if norm > max_step:
                    delta_m *= max_step / norm
                if norm > 0:
                    current = kin.fk(measured_q)
                    target = cart.Pose(current.xyz_m + delta_m, current.rotation.copy())
                    try:
                        hold_q = kin.plan_cartesian_line(
                            measured_q,
                            target,
                            orientation_weight=args.orientation_weight,
                            cartesian_step_m=max_step,
                        )[-1]
                    except Exception as exc:
                        hold_q = measured_q.copy()
                        print(f"\nIK rejected; holding: {exc}")
                wrist = cart.ARM_JOINTS.index("wrist_roll")
                hold_q[wrist] = np.clip(
                    hold_q[wrist]
                    + math.radians(args.wrist_roll_speed_deg_s) * dt * action["delta_wrist_roll"],
                    kin.lower[wrist],
                    kin.upper[wrist],
                )
                grip_direction = (
                    1.0 if action["gripper"] == GripperAction.OPEN
                    else -1.0 if action["gripper"] == GripperAction.CLOSE
                    else 0.0
                )
                hold_g = float(np.clip(
                    hold_g + grip_direction * args.gripper_speed_deg_s * dt,
                    *grip_limits,
                ))
                at_home = False

            sent = send_with_cached_safety_limit(
                driver, cart, hold_q, hold_g, obs, args.max_relative_target_deg
            )
            if action_callback is not None:
                action_callback(sent)
            hold_q, hold_g = sent_state(cart, sent)
            rate_cycles += 1
            rate_elapsed = time.monotonic() - rate_started
            if rate_elapsed >= 1.0:
                actual_hz = rate_cycles / rate_elapsed
                rate_started = time.monotonic()
                rate_cycles = 0
            if now - last_status >= 0.5:
                q_deg = cart.model_rad_to_lerobot_deg(measured_q)
                t_deg = cart.model_rad_to_lerobot_deg(hold_q)
                mode = (
                    "ESTOP" if estop else "DEPLOY" if deploy else
                    "HOME" if at_home else "LOCKED" if not operating_reached else
                    control_mode if enabled else f"HOLD-{control_mode}"
                )
                print(
                    "\r" + f"mode={mode:<6} q=[" + ",".join(f"{x:+.1f}" for x in q_deg)
                    + "] target=[" + ",".join(f"{x:+.1f}" for x in t_deg)
                    + f"] L=({state.left_x:+.2f},{state.left_y:+.2f})"
                    + f" R=({state.right_x:+.2f},{state.right_y:+.2f})"
                    + f" RB={int(state.rb)} hz={actual_hz:4.1f}".ljust(54),
                    end="", flush=True,
                )
                last_status = now
            last_rb, last_lb = state.rb, state.lb
            last_b, last_a, last_view = state.b, state.a, state.view
            last_left_stick = state.left_stick
            yield max(period - (time.monotonic() - started), 0.0)
    finally:
        teleop.disconnect()
        if driver is not None and driver.robot is not None and driver.robot.is_connected:
            if torque_holding:
                try:
                    obs = driver.observation()
                    send_with_cached_safety_limit(
                        driver,
                        cart,
                        driver.arm_q_model_rad(obs),
                        float(obs[cart.GRIPPER_KEY]),
                        obs,
                        args.max_relative_target_deg,
                    )
                except Exception as exc:
                    print(f"\nWARNING: final hold failed: {exc}")
                cart.wait_before_releasing_torque()
            driver.disconnect()


def command_control(args: argparse.Namespace) -> None:
    from contextlib import closing

    with closing(control_steps(args)) as steps:
        for remaining in steps:
            time.sleep(remaining)


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate(args)
        if args.command == "self-test":
            command_self_test(args)
        elif args.command == "capture-neutral":
            command_capture(args)
        else:
            command_control(args)
        return 0
    except (ControlError, XboxInputError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
