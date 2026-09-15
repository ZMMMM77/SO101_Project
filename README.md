# SO-101 Xbox Controller

Controller and diagnostic tools for operating an SO-101 follower arm from an
Xbox Series controller on macOS. The main controller supports direct joint
control, Cartesian XYZ control, a folded HOME pose, and a captured NEUTRAL
working pose.

## Current control behavior

- 20 Hz controller input and robot command loop
- One motor-bus position read and one synchronized six-motor write per cycle
- Quintic joint-space trajectory between HOME and NEUTRAL
- RB deadman switch and B-button software emergency stop
- Calibrated joint limits and per-cycle relative-target limiting
- Simultaneous multi-joint input

### JOINT mode

Hold `RB` while commanding motion.

| Input | Joint/action |
| --- | --- |
| Left stick left/right | Base / shoulder pan |
| Left stick up/down | Shoulder lift |
| Right stick up/down | Elbow flex |
| `Y` / `X` | Wrist flex in opposite directions |
| Right stick left/right | Wrist roll |
| `RT` / `LT` | Open / close gripper |

Click the left stick while `RB` is released to toggle JOINT and XYZ modes.
Hold `RB` and tap `LB` to move between HOME and NEUTRAL. Releasing `RB` during
the trajectory cancels the move and holds the measured pose.

## Project layout

```text
SO101_Project/
├── README.md
├── LICENSE
├── requirements-so101.txt
├── activate_so101.zsh
├── *.py
├── xbox_controller/
├── scripts/
├── simulation/
├── config/
└── .gitignore
```

## Installation

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-so101.txt
```

The checked-in `config/my_follower.json`, `folded_pose.json`, and
`neutral_pose.json` are hardware-specific. Do not apply this calibration to a
different arm without checking or recalibrating it first.

For the included `my_follower` calibration, create the local LeRobot path after
cloning:

```bash
mkdir -p .lerobot/calibration/robots/so_follower
ln -s ../../../../config/my_follower.json \
  .lerobot/calibration/robots/so_follower/my_follower.json
```

Find the current USB serial port with:

```bash
source .venv/bin/activate
lerobot-find-port
```

The launch scripts currently contain the original Mac USB port. Pass the
current port to the main launcher when it differs:

```bash
./xbox_controller/run_safe_control.sh --port /dev/cu.usbmodemYOUR_PORT
```

## Input-only test

Pair the Xbox controller in macOS Bluetooth settings, then run:

```bash
./xbox_controller/run_input_test.sh
```

This test does not connect to the robot.

## Offline controller test

```bash
.venv/bin/python xbox_controller/xbox_official_control.py self-test
.venv/bin/python xbox_controller/test_xbox_teleoperator.py
```

These tests do not connect to the controller or robot.

## Real-arm operation

Power the arm, support it during startup, keep the workspace clear, and run:

```bash
./xbox_controller/run_safe_control.sh --port /dev/cu.usbmodemYOUR_PORT
```

Wait until the current pose has been captured and held before releasing the
arm. Release `RB` immediately if motion is unexpected. Use `MENU` to exit.

## License

Apache License 2.0. See [LICENSE](LICENSE).

