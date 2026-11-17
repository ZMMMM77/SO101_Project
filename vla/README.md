# SO-101 SmolVLA workflow

The VMware Ubuntu guest owns the cameras and the robot. It records MCAP and converts demonstrations locally. WSL is the CUDA training environment. The Mac can preview the cameras using Foxglove.

## Official dependencies

- ROS 2 Jazzy provides `rosbag2`, `rosbag2_py`, and the MCAP storage plugin. The VMware guest uses the installed ROS packages.
- [Hugging Face LeRobot v0.5.1](https://github.com/huggingface/lerobot/releases/tag/v0.5.1) provides LeRobotDataset v3 and SmolVLA. The VMware project virtual environment uses the same version. The official checkout in `third_party/lerobot` is pinned to `1396b9fab7aecddd10006c33c47a487ffdcb54b4`. Recreate it from a fresh project checkout with:

```bash
git clone --depth 1 --branch v0.5.1 https://github.com/huggingface/lerobot.git third_party/lerobot
```

## Record on Ubuntu

Start the project environment from the project root, using the VMware checkout:

```bash
cd /home/zmmm/SO101_Project
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
```

With teleoperation running and both camera views confirmed, run this from a separate sourced terminal. Use one directory per demonstration. Press Ctrl+C in the recording terminal at the end of the episode.

```bash
python -m vla.record_episode data/raw/episode_0001
ros2 bag info data/raw/episode_0001
```

| ROS topic | Dataset field |
| --- | --- |
| `/so101/top/image_raw` | `observation.images.top` |
| `/so101/wrist/image_raw` | `observation.images.wrist` |
| `/so101/joint_states`, `/so101/gripper_position` | `observation.state` |
| `/so101/joint_targets`, `/so101/gripper_target` | `action` |

## Convert and inspect

```bash
python -m vla.convert_mcap data/raw/episode_0001 \
  --output data/lerobot/pick_place \
  --task "Pick up the block and put it in the box."
lerobot-dataset-viz --repo-id local/so101 --root data/lerobot/pick_place --episode-index 0
```

Specify multiple episode directories or MCAP files to build one dataset for the same task. The converter uses ROS bag receive timestamps for all six streams, preserves observation-before-action ordering, and samples at 20 Hz by default. It uses the project's joint ordering and calibration conversion to produce five joint values in calibrated degrees plus the calibrated gripper value for both state and action. It converts camera images to RGB and writes H.264 video using LeRobotDataset v3.

Review the camera roles and the alignment at gripper closure and object lift. This converter trims incomplete start and end frames and rejects a gap between valid frames. Pass `--fps` or `--max-skew-ms` to adjust the sampling cadence and observation age limit.

## Train on WSL

Copy the project code and the converted dataset to WSL. Use a fresh WSL virtual environment with CUDA-enabled PyTorch and install the official checkout with its SmolVLA dependencies.

```bash
nvidia-smi
python -m pip install -e './third_party/lerobot[smolvla]'
python -m vla.train_smolvla data/lerobot/pick_place \
  --output outputs/train/pick_place_smoke --steps 5 --batch-size 1
```

The training entry point calls the official LeRobot trainer and maps the top and wrist views to `camera1` and `camera2` for `lerobot/smolvla_base`. Training outputs are local. After a short run on the target GPU, choose a full training schedule and an unused output path.

A trained policy requires a ROS 2 action-execution interface before live control. The current follower consumes `/so101/joy` for Xbox control and publishes `/so101/joint_targets` for observation and recording.
