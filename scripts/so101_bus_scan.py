#!/usr/bin/env python3
"""Read-only SO-101 motor-bus diagnostic."""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


PORT = "/dev/cu.usbmodem5B3E0888091"
ROBOT_ID = "my_follower"


def main() -> None:
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    bus = robot.bus
    bus.connect(handshake=False)
    try:
        print(f"port={PORT}")
        print("motor          id  ping   voltage  temp  raw_position")
        for name, motor in bus.motors.items():
            model = bus.ping(name, num_retry=3)
            if model is None:
                print(f"{name:14s} {motor.id:2d}  FAIL")
                continue
            voltage = bus.read("Present_Voltage", name, normalize=False, num_retry=3)
            temp = bus.read("Present_Temperature", name, normalize=False, num_retry=3)
            pos = bus.read("Present_Position", name, normalize=False, num_retry=3)
            print(f"{name:14s} {motor.id:2d}  OK     {voltage:3d}      {temp:3d}   {pos:4d}")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
