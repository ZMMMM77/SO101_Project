# Xbox Teleoperation

Use the [project README](../README.md) to start the ROS 2 controller on
Ubuntu. The same controller logic also runs directly on macOS.

## Controls

Hold `RB` to move. Release it to hold position and cancel an active pose move.

| Input | JOINT mode | XYZ mode |
| --- | --- | --- |
| Left stick left/right | Shoulder pan | End-effector X |
| Left stick up/down | Shoulder lift | End-effector Y |
| Right stick up/down | Elbow flex | End-effector Z |
| Right stick left/right | Wrist roll | Wrist roll |
| `Y` / `X` | Wrist flex | |
| `RT` / `LT` | Open / close gripper | Open / close gripper |

- Hold `RB` and tap `LB` to move between HOME and NEUTRAL. Release `RB`
  once when the move finishes.
- Click the left stick with `RB` released to switch JOINT and XYZ modes.
- Press `B` to stop and hold. Press `A` with `RB` and `B` released to reset.
- Press `VIEW` for measured joints and FK, or `MENU` to exit. Support the arm
  before releasing torque.

HOME and NEUTRAL are stored in `folded_pose.json` and `neutral_pose.json`.
Ordinary joystick motion is enabled at NEUTRAL.

## Direct Python on macOS

From the checkout, install the dependencies and create the calibration link:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-so101.txt
mkdir -p .lerobot/calibration/robots/so_follower
ln -s ../../../../config/my_follower.json .lerobot/calibration/robots/so_follower/my_follower.json
```

Pair the controller over Bluetooth, then test input in a foreground terminal:

```bash
./xbox_controller/run_input_test.sh
lerobot-find-port
./xbox_controller/run_safe_control.sh --port /dev/cu.usbmodemYOUR_PORT
```

Run the last command with the actual port after checking input. The current
launcher runs at 20 Hz. The legacy sequential-unfold controller and its
return-pose options are available through
`ros2 run so101_bringup tool xbox-legacy --help`.
