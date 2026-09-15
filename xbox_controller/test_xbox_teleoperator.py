#!/usr/bin/env python3
"""Offline unit tests for the Xbox-to-LeRobot conversion layer."""

from __future__ import annotations

import unittest

from xbox_teleoperator import (
    GripperAction,
    XboxActionMapper,
    XboxState,
    apply_deadzone,
    normalize_signed_axis,
    normalize_trigger_axis,
)


class XboxMappingTests(unittest.TestCase):
    def test_signed_axis_endpoints(self) -> None:
        self.assertEqual(normalize_signed_axis(-32768), -1.0)
        self.assertEqual(normalize_signed_axis(0), 0.0)
        self.assertEqual(normalize_signed_axis(32767), 1.0)

    def test_trigger_encodings(self) -> None:
        self.assertEqual(normalize_trigger_axis(0), 0.0)
        self.assertEqual(normalize_trigger_axis(32767), 1.0)
        self.assertEqual(normalize_trigger_axis(-32768), 0.0)

    def test_deadzone(self) -> None:
        self.assertEqual(apply_deadzone(0.09, 0.10), 0.0)
        self.assertAlmostEqual(apply_deadzone(1.0, 0.10), 1.0)
        self.assertAlmostEqual(apply_deadzone(-1.0, 0.10), -1.0)

    def test_f710_compatible_xyz_mapping(self) -> None:
        mapper = XboxActionMapper(deadzone=0.0)
        action = mapper.map_action(
            XboxState(left_x=0.5, left_y=-0.25, right_x=-0.5, right_y=0.75)
        )
        self.assertEqual(action["delta_x"], -0.5)
        self.assertEqual(action["delta_y"], 0.25)
        self.assertEqual(action["delta_z"], -0.75)
        self.assertEqual(action["delta_wrist_roll"], 0.5)

    def test_gripper_mapping(self) -> None:
        mapper = XboxActionMapper(trigger_threshold=0.15)
        self.assertEqual(mapper.map_action(XboxState())["gripper"], GripperAction.STAY)
        self.assertEqual(
            mapper.map_action(XboxState(right_trigger=1.0))["gripper"],
            GripperAction.OPEN,
        )
        self.assertEqual(
            mapper.map_action(XboxState(left_trigger=1.0))["gripper"],
            GripperAction.CLOSE,
        )
        self.assertEqual(
            mapper.map_action(XboxState(left_trigger=1.0, right_trigger=1.0))["gripper"],
            GripperAction.STAY,
        )

    def test_safety_events(self) -> None:
        events = XboxActionMapper.map_events(XboxState(lb=True, rb=True, b=True, menu=True))
        self.assertTrue(events["is_intervention"])
        self.assertTrue(events["return_to_original"])
        self.assertTrue(events["emergency_stop"])
        self.assertTrue(events["quit"])


if __name__ == "__main__":
    unittest.main()
