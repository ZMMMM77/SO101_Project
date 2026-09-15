#!/usr/bin/env python3
"""User-editable original pose for the Xbox one-button return feature.

All values use the calibrated LeRobot degree convention printed by the
controller's VIEW/status output.  Keep values as ``None`` until a physically
safe pose has been measured and confirmed on the real arm.
"""

from __future__ import annotations


ORIGINAL_POSE_LEROBOT_DEG: dict[str, float | None] = {
    "shoulder_pan": None,
    "shoulder_lift": None,
    "elbow_flex": None,
    "wrist_flex": None,
    "wrist_roll": None,
    # Optional. Leave as None to keep the gripper at its current opening.
    "gripper": None,
}
