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

Prepare a separate WSL Python 3.12 environment at `~/SO101_Training/.venv`
with `torch==2.10.0+cu126`, `torchvision==0.25.0+cu126`,
`torchcodec==0.10.0`, and `lerobot[smolvla]==0.5.1`.
The training script selects PyAV for video decoding.

VMware retains the project and datasets. Set `VM` to the VMware SSH account
and reachable address, then mount its project read-only from WSL.
Install `sshfs` and `rsync` in WSL if needed.

```bash
VM="<VM_USER>@<VM_HOST>"
mkdir -p ~/mnt/SO101_Project
sshfs -o ro,reconnect,ServerAliveInterval=15 "${VM}:SO101_Project" ~/mnt/SO101_Project
```

Activate the WSL environment and keep caches and outputs on its local filesystem.

```bash
source ~/SO101_Training/.venv/bin/activate
export HF_HOME="$HOME/SO101_Training/.hf-home"
export HF_LEROBOT_HOME="$HOME/SO101_Training/.lerobot"
cd ~/mnt/SO101_Project
python -m vla.train_smolvla data/lerobot/pick_place \
  --output "$HOME/SO101_Training/outputs/pick_place_smoke" --steps 5 --batch-size 1
```

The trainer loads `lerobot/smolvla_base` and maps the top and wrist views to
`camera1` and `camera2`. Use a new output directory for each run. After a short
test, measure sustained training speed and GPU memory before choosing a full schedule.

Copy the complete run back to VMware, including the optimizer state and processors.

```bash
ssh "$VM" 'mkdir -p ~/SO101_Project/outputs'
rsync -a --info=progress2 "$HOME/SO101_Training/outputs/pick_place_smoke" "${VM}:SO101_Project/outputs/"
```

A five-step test with the official `lerobot/svla_so100_pickplace` dataset
passed on the 6 GB RTX 3060 Laptop GPU. The returned model's SHA-256 matched
the WSL copy. Custom demonstrations still require capture and validation.

A trained policy requires a ROS 2 action-execution interface before live control.
The current follower consumes `/so101/joy` for Xbox control and publishes
`/so101/joint_targets` for observation and recording.
