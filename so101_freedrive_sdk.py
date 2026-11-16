#!/usr/bin/env python3
"""Safe passive freedrive / leader SDK for an SO-101 arm.

This module wraps LeRobot's ``SO101Leader`` implementation.  On the STS3215
hardware used by SO-101, the supported hand-guiding mode is passive freedrive:
motor torque is disabled and calibrated joint positions are continuously read.

This is deliberately *not* advertised as active gravity compensation.  The
arm can fall under gravity while torque is disabled, so support it by hand
before enabling freedrive or releasing a held pose.

Typical SDK use::

    from so101_freedrive_sdk import SO101FreedriveSDK

    arm = SO101FreedriveSDK()
    arm.connect()                 # Opens serial; does not change torque.
    arm.enable_freedrive()        # Disables torque: support the arm first.
    action = arm.get_action()     # LeRobot-compatible {"joint.pos": value}.
    arm.hold_current_pose()       # Position servo holds the current pose.
    arm.release()                 # Support first; disables torque again.
    arm.disconnect()

The command-line interface can perform an offline self-test, read one pose,
run freedrive, or record the dragged trajectory to CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from lerobot.motors.feetech import OperatingMode
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


DEFAULT_PORT = "/dev/cu.usbmodem5B3E0888091"
DEFAULT_ROBOT_ID = "my_follower"
DEFAULT_CALIBRATION_DIR = Path(
    __file__
).resolve().parent / ".lerobot" / "calibration" / "robots" / "so_follower"

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_KEYS = tuple(f"{joint}.pos" for joint in JOINTS)


class FreedriveError(RuntimeError):
    """Base error for the freedrive wrapper."""


class CalibrationError(FreedriveError):
    """The selected calibration is missing, invalid, or mismatched."""


@dataclass(frozen=True)
class FreedriveSample:
    """One timestamped, calibrated leader-arm action sample."""

    monotonic_time_s: float
    wall_time_s: float
    action: dict[str, float]


class SO101FreedriveSDK:
    """LeRobot-compatible passive freedrive wrapper for one SO-101 arm.

    ``connect`` only opens the serial bus and verifies calibration.  It does
    not change motor torque.  Call ``enable_freedrive`` explicitly to disable
    torque.  This two-stage API avoids an unexpected drop merely from opening
    the serial connection.
    """

    def __init__(
        self,
        *,
        port: str = DEFAULT_PORT,
        robot_id: str = DEFAULT_ROBOT_ID,
        calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR,
        frequency_hz: float = 30.0,
    ) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be greater than zero")

        self.port = port
        self.robot_id = robot_id
        self.calibration_dir = Path(calibration_dir).expanduser()
        self.frequency_hz = float(frequency_hz)
        self._leader: SO101Leader | None = None
        self._freedrive_enabled = False
        self._holding = False

    @property
    def calibration_path(self) -> Path:
        return self.calibration_dir / f"{self.robot_id}.json"

    @property
    def is_connected(self) -> bool:
        return self._leader is not None and self._leader.is_connected

    @property
    def is_freedrive_enabled(self) -> bool:
        return self._freedrive_enabled

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def action_features(self) -> dict[str, type]:
        """Feature schema expected by LeRobot teleoperation code."""
        if self._leader is not None:
            return self._leader.action_features
        return {key: float for key in ACTION_KEYS}

    @property
    def teleoperator(self) -> SO101Leader:
        """Return the wrapped LeRobot teleoperator instance."""
        self._require_connected()
        assert self._leader is not None
        return self._leader

    def validate_calibration_file(self) -> dict[str, dict[str, int]]:
        """Validate the saved follower calibration without touching hardware."""
        path = self.calibration_path
        if not path.is_file():
            raise CalibrationError(
                f"Calibration file not found: {path}\n"
                "Do not recalibrate blindly. Check --robot-id and --calibration-dir first."
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"Cannot read calibration file {path}: {exc}") from exc

        missing = [joint for joint in JOINTS if joint not in data]
        if missing:
            raise CalibrationError(f"Calibration is missing joints: {', '.join(missing)}")

        required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
        for joint in JOINTS:
            entry = data[joint]
            if not isinstance(entry, dict):
                raise CalibrationError(f"Calibration entry '{joint}' is not an object")
            absent = required.difference(entry)
            if absent:
                raise CalibrationError(
                    f"Calibration entry '{joint}' is missing: {', '.join(sorted(absent))}"
                )
            if int(entry["range_min"]) >= int(entry["range_max"]):
                raise CalibrationError(f"Invalid range for '{joint}'")

        ids = [int(data[joint]["id"]) for joint in JOINTS]
        if ids != [1, 2, 3, 4, 5, 6]:
            raise CalibrationError(f"Expected motor IDs 1..6, found {ids}")
        return data

    def connect(self) -> None:
        """Open the serial bus and verify calibration; do not alter torque."""
        if self.is_connected:
            return

        self.validate_calibration_file()
        config = SO101LeaderConfig(
            port=self.port,
            id=self.robot_id,
            calibration_dir=self.calibration_dir,
            use_degrees=True,
        )
        leader = SO101Leader(config)

        try:
            # SO101Leader.connect() immediately disables torque in configure().
            # Opening the underlying bus directly lets the caller explicitly
            # choose when the arm is safe to release.
            leader.bus.connect()
            if not leader.is_calibrated:
                raise CalibrationError(
                    "The saved calibration does not match the limits/offsets stored in the motors. "
                    "No calibration was written. Verify that this is the same arm and robot ID."
                )
        except Exception:
            if leader.bus.is_connected:
                # Preserve the pre-existing torque state on failed validation.
                leader.bus.disconnect(disable_torque=False)
            raise

        self._leader = leader
        self._freedrive_enabled = False
        self._holding = False

    def enable_freedrive(self) -> None:
        """Disable all motor torque so an operator can drag the arm by hand.

        Support the arm before calling this method.  This mode has no active
        gravity compensation; the arm may fall immediately under gravity.
        """
        self._require_connected()
        assert self._leader is not None
        # Mark first so cleanup attempts a safe hold even if a multi-motor
        # torque-disable operation fails part-way through the chain.
        self._freedrive_enabled = True
        self._holding = False
        self._leader.bus.disable_torque(num_retry=2)

    def get_action(self) -> dict[str, float]:
        """Return calibrated joints in LeRobot leader-action format."""
        self._require_connected()
        assert self._leader is not None
        action = self._leader.get_action()
        from so101_events import report_position

        return report_position({key: float(action[key]) for key in ACTION_KEYS})

    def read_joints(self) -> dict[str, float]:
        """Return joint names without the LeRobot ``.pos`` suffix."""
        return {key.removesuffix(".pos"): value for key, value in self.get_action().items()}

    def sample(self) -> FreedriveSample:
        """Capture one timestamped action sample."""
        return FreedriveSample(
            monotonic_time_s=time.monotonic(),
            wall_time_s=time.time(),
            action=self.get_action(),
        )

    def stream(self, duration_s: float | None = None) -> Iterator[FreedriveSample]:
        """Yield samples near ``frequency_hz`` until duration expires."""
        self._require_connected()
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")

        period_s = 1.0 / self.frequency_hz
        started = time.monotonic()
        deadline = started
        while duration_s is None or time.monotonic() - started < duration_s:
            yield self.sample()
            deadline += period_s
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Avoid trying to catch up with a burst after a slow serial read.
                deadline = time.monotonic()

    def hold_current_pose(self) -> dict[str, float]:
        """Enable the position servos at the measured pose, without moving home.

        Goal positions are written while torque is off, then torque is enabled.
        This minimizes a jump at engagement.  It is ordinary position holding,
        not model-based gravity compensation.
        """
        self._require_connected()
        assert self._leader is not None
        bus = self._leader.bus

        current = bus.sync_read("Present_Position")
        bus.disable_torque(num_retry=2)
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=2)
        bus.sync_write("Goal_Position", current, num_retry=2)
        bus.enable_torque(num_retry=2)
        # Send once more after torque-on in case a servo ignored the first goal.
        bus.sync_write("Goal_Position", current, num_retry=2)

        self._freedrive_enabled = False
        self._holding = True
        return {f"{joint}.pos": float(value) for joint, value in current.items()}

    def release(self) -> None:
        """Disable torque. Support the arm first because it may drop."""
        self._require_connected()
        assert self._leader is not None
        self._freedrive_enabled = True
        self._holding = False
        self._leader.bus.disable_torque(num_retry=2)

    def disconnect(self, *, release_torque: bool = True) -> None:
        """Close serial; normally disable torque before closing.

        ``release_torque=False`` preserves the current hardware torque state.
        This can leave the arm energized without software supervision and is
        only intended for controlled integration code.
        """
        if not self.is_connected:
            self._leader = None
            return

        assert self._leader is not None
        self._leader.bus.disconnect(disable_torque=release_torque)
        self._leader = None
        self._freedrive_enabled = False
        self._holding = False

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise FreedriveError("Arm is not connected. Call connect() first.")


def _print_action(action: dict[str, float], *, file: TextIO = sys.stdout) -> None:
    values = "  ".join(f"{key.removesuffix('.pos')}={value:8.2f}" for key, value in action.items())
    print(values, file=file)


def _confirm_freedrive(skip_confirmation: bool) -> None:
    if skip_confirmation:
        return
    print("\nDANGER: enabling freedrive disables all motor torque.")
    print("The arm has NO active gravity compensation and may fall immediately.")
    print("Support the arm with one hand and keep clear of pinch points.")
    answer = input("Type FREE to continue: ").strip()
    if answer != "FREE":
        raise FreedriveError("Cancelled: confirmation was not exactly 'FREE'.")


def _safe_finish(arm: SO101FreedriveSDK) -> None:
    """Hold first, then explicitly release and disconnect."""
    if not arm.is_connected:
        return

    if not arm.is_freedrive_enabled and not arm.is_holding:
        # Connection/setup failed before we intentionally touched torque.
        arm.disconnect(release_torque=False)
        return

    if arm.is_freedrive_enabled:
        try:
            arm.hold_current_pose()
            print("\nCurrent pose is now HELD by the position servos.")
        except Exception as exc:
            print(f"\nWARNING: could not engage hold: {exc}", file=sys.stderr)
            print("Keep supporting the arm and use the power switch if needed.", file=sys.stderr)

    if arm.is_holding:
        try:
            input("Support the arm, then press ENTER to release torque and disconnect. ")
        except EOFError:
            print("\nNo terminal input available; leaving torque enabled while disconnecting.")
            arm.disconnect(release_torque=False)
            return

    arm.disconnect(release_torque=True)
    print("Torque released and serial connection closed.")


def _make_arm(args: argparse.Namespace) -> SO101FreedriveSDK:
    return SO101FreedriveSDK(
        port=args.port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        frequency_hz=args.frequency_hz,
    )


def _cmd_self_test(args: argparse.Namespace) -> int:
    arm = _make_arm(args)
    calibration = arm.validate_calibration_file()
    version = importlib.metadata.version("lerobot")
    config = SO101LeaderConfig(
        port=args.port,
        id=args.robot_id,
        calibration_dir=Path(args.calibration_dir),
        use_degrees=True,
    )
    leader = SO101Leader(config)
    if tuple(leader.action_features) != ACTION_KEYS:
        raise FreedriveError(f"Unexpected LeRobot action schema: {leader.action_features}")

    print("Offline self-test passed. No robot connection was made.")
    print(f"LeRobot version: {version}")
    print(f"Port setting:     {args.port}")
    print(f"Robot ID:         {args.robot_id}")
    print(f"Calibration:      {arm.calibration_path}")
    print(f"Motor IDs:        {[calibration[joint]['id'] for joint in JOINTS]}")
    print("Mode: passive torque-off freedrive (not active gravity compensation)")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    arm = _make_arm(args)
    try:
        arm.connect()
        print("Connected and calibration matched. Torque state was not changed.")
        _print_action(arm.get_action())
    finally:
        # Preserve torque state: this command is read-only.
        arm.disconnect(release_torque=False)
    return 0


def _run_freedrive(args: argparse.Namespace, output_path: Path | None) -> int:
    _confirm_freedrive(args.yes)
    arm = _make_arm(args)
    output_file: TextIO | None = None
    writer: csv.DictWriter | None = None
    sample_count = 0
    last_print = 0.0

    try:
        arm.connect()
        print(f"Connected to {args.port}; calibration matched.")
        arm.enable_freedrive()
        print("FREEDRIVE ACTIVE: torque is OFF. Drag the arm while supporting its weight.")
        print("Press Ctrl-C to stop; the script will first hold the current pose.")

        if output_path is not None:
            output_path = output_path.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path.open("w", newline="", encoding="utf-8")
            fieldnames = ["sample", "wall_time_s", "elapsed_s", *ACTION_KEYS]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            print(f"Recording CSV: {output_path}")

        started = time.monotonic()
        for sample in arm.stream(args.duration_s):
            elapsed = sample.monotonic_time_s - started
            if writer is not None:
                writer.writerow(
                    {
                        "sample": sample_count,
                        "wall_time_s": f"{sample.wall_time_s:.9f}",
                        "elapsed_s": f"{elapsed:.9f}",
                        **sample.action,
                    }
                )
            sample_count += 1

            if sample.monotonic_time_s - last_print >= 1.0 / args.print_hz:
                _print_action(sample.action)
                last_print = sample.monotonic_time_s
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        if output_file is not None:
            output_file.flush()
            output_file.close()
        _safe_finish(arm)

    if output_path is not None:
        print(f"Saved {sample_count} samples to {output_path}")
    return 0


def _cmd_freedrive(args: argparse.Namespace) -> int:
    return _run_freedrive(args, output_path=None)


def _cmd_record(args: argparse.Namespace) -> int:
    return _run_freedrive(args, output_path=Path(args.output))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 passive freedrive / LeRobot leader SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--frequency-hz", type=float, default=30.0)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="offline imports/config/calibration check")
    subparsers.add_parser("read", help="read one joint pose without changing torque")

    def add_motion_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--duration-s", type=float, default=None)
        command.add_argument("--print-hz", type=float, default=2.0)
        command.add_argument(
            "--yes",
            action="store_true",
            help="skip the typed FREE confirmation (unsafe for interactive use)",
        )

    freedrive = subparsers.add_parser("freedrive", help="torque-off drag mode; print joint actions")
    add_motion_options(freedrive)

    record = subparsers.add_parser("record", help="torque-off drag mode; save joint actions to CSV")
    add_motion_options(record)
    record.add_argument("--output", required=True, help="CSV output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "print_hz", 1.0) <= 0:
        parser.error("--print-hz must be greater than zero")

    commands = {
        "self-test": _cmd_self_test,
        "read": _cmd_read,
        "freedrive": _cmd_freedrive,
        "record": _cmd_record,
    }
    try:
        return commands[args.command](args)
    except (FreedriveError, CalibrationError, ConnectionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
