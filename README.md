# SO-101 ROS 2 and SmolVLA

SO-101 follower control with Xbox joint/XYZ teleoperation, two USB cameras,
diagnostics, and a SmolVLA training pipeline.

VMware Ubuntu handles hardware, recording, and dataset storage. WSL Ubuntu
reads the project through SSHFS and trains on the RTX 3060. Training outputs
are copied back to VMware. The Mac provides VS Code Remote-SSH and Foxglove.

## Project layout

| Path | Purpose |
| --- | --- |
| `ros2/so101_bringup/` | ROS nodes and launch files |
| `xbox_controller/`, `scripts/` | Teleoperation and hardware utilities |
| `config/`, `simulation/` | Calibration, saved poses, and URDF |
| `vla/` | MCAP recording, LeRobot conversion, and SmolVLA training |
| `data/raw/`, `data/lerobot/` | ROS recordings and converted datasets |
| `third_party/lerobot/` | Pinned official LeRobot source |
| `outputs/` | Returned checkpoints and training outputs |

## Setup

Install ROS 2 Jazzy, colcon, and Python 3.12 with venv on VMware Ubuntu 24.04.
Run from the project root.
[ROS installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

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

Use your arm's calibration and saved poses. Attach both cameras, the arm,
and the Bluetooth adapter to the VMware guest. For serial access, add your
user to `dialout` with `sudo usermod -aG dialout "$USER"`, then sign in again.

## Run

Source the environment in each new VMware terminal.

```bash
cd ~/SO101_Project
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 launch so101_bringup so101.launch.py
```

Motion is disabled by default. Two InnoMaker U20CAM-1080P-S1 cameras are assigned
`top` then `wrist` in USB path order. Check the views after reconnecting devices and override them with `top_device` and `wrist_device`. Topics are
`/so101/top/image_raw` and `/so101/wrist/image_raw`.

After checking the controller and port, stop the preview, support the arm,
and release the sticks, triggers, and `RB` before starting live control.

```bash
ros2 launch so101_bringup so101.launch.py enable_motion:=true port:=/dev/ttyACM0
```

Hold `RB` to move. Release `RB` and click the left stick to switch JOINT/XYZ.
Support the arm before stopping.
[Xbox controls](xbox_controller/README.md)

## VLA workflow

Record MCAP on VMware, convert to LeRobotDataset v3 at 20 Hz, then fine-tune
`lerobot/smolvla_base` on WSL. The trainer maps `top` and `wrist` to the model's
camera inputs.

```bash
python -m vla.record_episode data/raw/episode_0001
# Stop recording with Ctrl+C, then convert.
python -m vla.convert_mcap data/raw/episode_0001 --output data/lerobot/pick_place --task "Pick up the block and put it in the box."
```

From the mounted project in the prepared WSL environment, test five steps.

```bash
python -m vla.train_smolvla data/lerobot/pick_place --output "$HOME/SO101_Training/outputs/pick_place_smoke" --steps 5 --batch-size 1
```

See the [VLA guide](vla/README.md) for environment versions, SSHFS mounts,
topic mappings, and checkpoint transfer.
Official references are the [LeRobot installation guide](https://huggingface.co/docs/lerobot/main/en/installation)
and [SmolVLA tutorial](https://huggingface.co/docs/lerobot/main/en/smolvla).

## ROS tools

Stop processes using the serial port before running diagnostics.

| Task | Command |
| --- | --- |
| Xbox offline test | `ros2 run so101_bringup tool xbox self-test` |
| Read-only motor scan | `ros2 run so101_bringup tool scan` |
| Joint test | `ros2 run so101_bringup tool joint-test --help` |
| Cartesian control | `ros2 run so101_bringup tool cartesian --help` |
| iPhone teleoperation | `ros2 run so101_bringup tool iphone --help` |
| Freedrive and CSV | `ros2 run so101_bringup tool freedrive --help` |

## Validation

Five-step fine-tuning on `lerobot/svla_so100_pickplace` passed with a 6 GB
RTX 3060 Laptop GPU, LeRobot 0.5.1, and PyTorch 2.10.0+cu126. The returned
model matched the WSL checkpoint's SHA-256.

Live ROS teleoperation and custom dual-camera demonstrations require hardware
validation. ROS 2 execution of trained policies is a further integration step.

[Apache License 2.0](LICENSE)
