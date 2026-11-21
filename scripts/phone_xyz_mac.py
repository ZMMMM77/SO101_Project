#!/usr/bin/env python3
"""Read HEBI Mobile I/O on the Mac and stream phone input over SSH."""

import argparse
import contextlib
import ipaddress
import json
import math
import shlex
import subprocess
import sys
import time


def find_phone(address):
    with contextlib.redirect_stdout(sys.stderr):
        import hebi
    for addresses in ([address], []):
        lookup = hebi.Lookup(addresses)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            for entry in lookup.entrylist:
                if str(entry.ip_address) == address and not entry.is_stale:
                    group = lookup.get_group_from_names(
                        [entry.family], [entry.name], timeout_ms=2000
                    )
                    if group is not None:
                        return lookup, group
            time.sleep(0.2)
    raise SystemExit("No HEBI reply. Keep Mobile I/O open and the phone screen awake.")


def stream_phone(group, sink, duration):
    group.feedback_frequency = 20
    start = time.monotonic()
    last_packet = start
    previous_b1 = False
    clutch = 0
    sequence = 0
    while duration == 0 or time.monotonic() - start < duration:
        feedback = group.get_next_feedback(timeout_ms=100)
        if feedback is None:
            if time.monotonic() - last_packet > 2:
                raise SystemExit("Phone feedback stopped.")
            continue
        sample = feedback[0]
        raw_position = sample.ar_position
        raw_orientation = sample.ar_orientation
        position = [] if raw_position is None else list(map(float, raw_position))
        quaternion = [] if raw_orientation is None else list(map(float, raw_orientation))
        norm = math.hypot(*quaternion)
        if not (
            len(position) == 3 and len(quaternion) == 4
            and all(map(math.isfinite, position + quaternion))
            and 0.9 < norm < 1.1
        ):
            if time.monotonic() - last_packet > 2:
                raise SystemExit("No valid ARKit tracking. Check the phone camera permission and lighting.")
            continue

        io = sample.io
        b1 = bool(io.b.get_int(1)) if io.b.has_int(1) else (
            bool(io.b.get_bool(1)) if hasattr(io.b, "has_bool") and io.b.has_bool(1) else False
        )
        a3 = float(io.a.get_float(3)) if io.a.has_float(3) else 0.0
        if not math.isfinite(a3):
            a3 = 0.0
        if b1 and not previous_b1:
            clutch += 1
        packet = {
            "sequence": sequence, "time": time.time(), "clutch": clutch,
            "position": position, "orientation_wxyz": quaternion,
            "b1": b1, "a3": max(-1.0, min(1.0, a3)),
        }
        sink.write(json.dumps(packet, allow_nan=False) + "\n")
        sink.flush()
        previous_b1 = b1
        sequence += 1
        last_packet = time.monotonic()
    print(f"Mac sent {sequence} phone samples.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-ip", required=True, type=lambda s: str(ipaddress.IPv4Address(s)))
    parser.add_argument("--host", default="zmmm@100.115.51.92")
    parser.add_argument("--project", default="/home/zmmm/SO101_Project")
    parser.add_argument("--seconds", default=0.0, type=float, help="0 runs until Ctrl+C")
    parser.add_argument("--enable-motion", action="store_true", help="Connect and control the arm")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if not args.host or args.host.startswith("-"):
        parser.error("--host must be an SSH host name or user@host")

    # Retain the Lookup while its feedback group is in use.
    lookup, group = find_phone(args.phone_ip)
    command = [".venv/bin/python", "-u", "scripts/phone_xyz_receiver.py"]
    if args.enable_motion:
        command.append("--enable-motion")
    remote = (
        f"cd {shlex.quote(args.project)} && source /opt/ros/jazzy/setup.bash"
        f" && exec {shlex.join(command)}"
    )
    ssh = subprocess.Popen(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2", args.host, remote],
        stdin=subprocess.PIPE, text=True, start_new_session=True,
    )
    print("Hold the phone screen up, with its top edge pointed toward the work area.", file=sys.stderr)
    print("Release B1 to arm, then hold B1 to follow translation. A3 controls the gripper.", file=sys.stderr)
    try:
        stream_phone(group, ssh.stdin, args.seconds)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        print("Ubuntu closed the phone input stream.", file=sys.stderr)
    finally:
        try:
            ssh.stdin.close()
        except BrokenPipeError:
            pass
        try:
            status = ssh.wait(timeout=8)
        except subprocess.TimeoutExpired:
            ssh.terminate()
            status = ssh.wait(timeout=3)
    return status


if __name__ == "__main__":
    sys.exit(main())
