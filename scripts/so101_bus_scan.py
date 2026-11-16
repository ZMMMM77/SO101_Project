#!/usr/bin/env python3
"""Read-only SO-101 motor-bus diagnostic."""

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


PORT = "/dev/cu.usbmodem5B3E0888091"
ROBOT_ID = "my_follower"


def main(on_result=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--robot-id", default=ROBOT_ID)
    args = parser.parse_args()
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    bus = robot.bus
    bus.connect(handshake=False)
    try:
        print(f"port={args.port}")
        print("motor          id  ping   voltage  temp  raw_position")
        for name, motor in bus.motors.items():
            model = bus.ping(name, num_retry=3)
            if model is None:
                print(f"{name:14s} {motor.id:2d}  FAIL")
                if on_result is not None:
                    on_result(name, {"id": motor.id, "online": False})
                continue
            voltage = bus.read("Present_Voltage", name, normalize=False, num_retry=3)
            temp = bus.read("Present_Temperature", name, normalize=False, num_retry=3)
            pos = bus.read("Present_Position", name, normalize=False, num_retry=3)
            print(f"{name:14s} {motor.id:2d}  OK     {voltage:3d}      {temp:3d}   {pos:4d}")
            if on_result is not None:
                on_result(name, {"id": motor.id, "online": True, "voltage_v": voltage / 10,
                                 "temperature_c": temp, "raw_position": pos})
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
