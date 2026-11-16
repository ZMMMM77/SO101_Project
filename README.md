# SO-101 ROS 2 Control

ROS 2 Jazzy control for an SO-101 follower arm, with Xbox teleoperation,
wrist and overhead cameras, and motor diagnostics.

## Setup

On Ubuntu 24.04, install [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html),
`colcon`, and Python 3.12 with `venv`. From the checkout, run:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements-ros2.txt
mkdir -p .lerobot/calibration/robots/so_follower
ln -s ../../../../config/my_follower.json .lerobot/calibration/robots/so_follower/my_follower.json
colcon build --base-paths ros2 --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

Use the calibration and recorded poses for your arm. Connect the arm, both
cameras, and the Xbox controller to Ubuntu. On VMware, attach the USB
devices or the Bluetooth adapter to the guest. If your serial port belongs
to `dialout`, run `sudo usermod -aG dialout "$USER"` and sign in again.

## Run

In each new terminal, source the ROS installation and the project environment.

```bash
cd ~/SO101_Project
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 launch so101_bringup so101.launch.py
```

Motion is disabled by default. Use the launch arguments `wrist_device` and
`top_device` to match the physical cameras. Images are published on
`/so101/wrist/image_raw` and `/so101/top/image_raw`.

After verifying the controller and serial port, stop the preview launch,
support the arm, release the sticks, triggers, and `RB`, then run:

```bash
ros2 launch so101_bringup so101.launch.py enable_motion:=true port:=/dev/ttyACM0
```

Support the arm before stopping the live launch. See the
[Xbox guide](xbox_controller/README.md) for the controls and macOS setup.

## ROS tools

Run diagnostics in a sourced terminal after stopping any live process that
uses the same serial port.

| Task | Command |
| --- | --- |
| Xbox offline test | `ros2 run so101_bringup tool xbox self-test` |
| Read-only motor scan | `ros2 run so101_bringup tool scan` |
| Joint test | `ros2 run so101_bringup tool joint-test --help` |
| Cartesian control | `ros2 run so101_bringup tool cartesian --help` |
| iPhone teleoperation | `ros2 run so101_bringup tool iphone --help` |
| Freedrive and CSV | `ros2 run so101_bringup tool freedrive --help` |

The Jazzy build, simultaneous 640×360 camera streams, and offline Xbox
safety logic were verified on VMware Ubuntu. Live ROS teleoperation still
requires hardware validation.

[Apache License 2.0](LICENSE)
