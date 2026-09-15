#!/usr/bin/env python3
"""Xbox Series X|S controller adapter with a LeRobot-compatible action shape.

The hardware backend uses SDL through pygame. SDL presents supported Xbox
controllers with stable, named axes/buttons, so this module does not depend on
controller-specific raw HID byte offsets.

The public action returned by :class:`XboxLeRobotTeleop` mirrors LeRobot's
gamepad teleoperator:

    {"delta_x": float, "delta_y": float, "delta_z": float,
     "delta_wrist_roll": float, "gripper": int}

No robot connection is made by this module.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class XboxInputError(RuntimeError):
    """Raised when the Xbox controller cannot be initialized or read."""


class GripperAction(IntEnum):
    """Same discrete convention used by LeRobot's gamepad teleoperator."""

    CLOSE = 0
    STAY = 1
    OPEN = 2


@dataclass(frozen=True)
class XboxState:
    """One normalized Xbox controller sample."""

    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    lb: bool = False
    rb: bool = False
    view: bool = False
    menu: bool = False
    left_stick: bool = False
    right_stick: bool = False
    timestamp: float = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_signed_axis(raw: int) -> float:
    """Convert an SDL signed axis value to [-1, 1]."""

    denominator = 32768.0 if raw < 0 else 32767.0
    return clamp(float(raw) / denominator, -1.0, 1.0)


def normalize_trigger_axis(raw: int) -> float:
    """Convert common SDL trigger encodings to [0, 1].

    SDL normally reports controller triggers as 0..32767. Some backends expose
    -32768..32767 instead, so both representations are accepted.
    """

    if raw < 0:
        return clamp((float(raw) + 32768.0) / 65535.0, 0.0, 1.0)
    return clamp(float(raw) / 32767.0, 0.0, 1.0)


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply and rescale a radial component deadzone."""

    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be within [0, 1).")
    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


class XboxActionMapper:
    """Map normalized Xbox state to the current LeRobot F710 action contract."""

    def __init__(self, deadzone: float = 0.10, trigger_threshold: float = 0.15):
        if not 0.0 <= trigger_threshold <= 1.0:
            raise ValueError("trigger_threshold must be within [0, 1].")
        self.deadzone = deadzone
        self.trigger_threshold = trigger_threshold

    def map_action(self, state: XboxState) -> dict[str, float | int]:
        left_x = apply_deadzone(state.left_x, self.deadzone)
        left_y = apply_deadzone(state.left_y, self.deadzone)
        right_x = apply_deadzone(state.right_x, self.deadzone)
        right_y = apply_deadzone(state.right_y, self.deadzone)

        # Match the axis convention in LeRobot's F710/RumblePad HID adapter.
        delta_x = -left_x
        delta_y = -left_y
        delta_z = -right_y
        delta_wrist_roll = -right_x

        open_requested = state.right_trigger >= self.trigger_threshold
        close_requested = state.left_trigger >= self.trigger_threshold
        if open_requested == close_requested:
            gripper = GripperAction.STAY
        elif open_requested:
            gripper = GripperAction.OPEN
        else:
            gripper = GripperAction.CLOSE

        return {
            "delta_x": float(delta_x),
            "delta_y": float(delta_y),
            "delta_z": float(delta_z),
            "delta_wrist_roll": float(delta_wrist_roll),
            "gripper": int(gripper),
        }

    @staticmethod
    def map_events(state: XboxState) -> dict[str, bool]:
        return {
            "is_intervention": state.rb,
            "return_to_original": state.lb,
            "emergency_stop": state.b,
            "reset_emergency_stop": state.a,
            "terminate_episode": state.x,
            "success": state.y,
            "rerecord_episode": state.a,
            "print_status": state.view,
            "quit": state.menu,
        }


class PygameXboxBackend:
    """SDL GameController backend for Xbox Wireless Controller."""

    def __init__(
        self,
        controller_index: int = 0,
        require_xbox_name: bool = True,
        connect_timeout_s: float = 10.0,
    ):
        self.controller_index = controller_index
        self.require_xbox_name = require_xbox_name
        self.connect_timeout_s = connect_timeout_s
        self.pygame: Any = None
        self.controller_module: Any = None
        self.controller: Any = None
        self.name = ""

    def connect(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        try:
            import pygame
            from pygame._sdl2 import controller as controller_module
        except ImportError as exc:
            raise XboxInputError(
                "pygame is not installed. Activate the project-local .venv "
                "and run: python -m pip install 'pygame>=2.6,<3'"
            ) from exc

        self.pygame = pygame
        self.controller_module = controller_module
        pygame.init()
        controller_module.init()

        deadline = time.monotonic() + self.connect_timeout_s
        count = controller_module.get_count()
        if count <= self.controller_index:
            print(
                f"Waiting up to {self.connect_timeout_s:.0f}s for SDL to rediscover the controller...",
                flush=True,
            )
        while count <= self.controller_index and time.monotonic() < deadline:
            pygame.event.pump()
            time.sleep(0.25)
            # macOS may briefly remove the synthetic SDL controller after the
            # previous process closes it. Reinitializing makes SDL rescan.
            controller_module.quit()
            controller_module.init()
            count = controller_module.get_count()

        if count <= self.controller_index:
            self.disconnect()
            raise XboxInputError(
                f"No SDL game controller at index {self.controller_index} after "
                f"waiting {self.connect_timeout_s:.0f}s; detected {count}. "
                "Press the Xbox button to wake the controller, then retry."
            )
        if not controller_module.is_controller(self.controller_index):
            self.disconnect()
            raise XboxInputError(
                f"Device index {self.controller_index} is not recognized as an SDL GameController."
            )

        self.controller = controller_module.Controller(self.controller_index)
        self.name = str(controller_module.name_forindex(self.controller_index) or "")
        if self.require_xbox_name and "xbox" not in self.name.lower():
            detected = self.name or "<unnamed controller>"
            self.disconnect()
            raise XboxInputError(
                f"Detected '{detected}', not an Xbox controller. "
                "Use --allow-non-xbox only when intentionally testing another SDL controller."
            )

    @property
    def is_connected(self) -> bool:
        return self.controller is not None and bool(self.controller.attached())

    def _axis(self, constant_name: str, *, trigger: bool = False) -> float:
        constant = getattr(self.pygame, constant_name)
        raw = int(self.controller.get_axis(constant))
        return normalize_trigger_axis(raw) if trigger else normalize_signed_axis(raw)

    def _button(self, constant_name: str) -> bool:
        constant = getattr(self.pygame, constant_name)
        return bool(self.controller.get_button(constant))

    def read(self) -> XboxState:
        if not self.is_connected:
            raise XboxInputError("Xbox controller is not connected.")
        self.pygame.event.pump()
        return XboxState(
            left_x=self._axis("CONTROLLER_AXIS_LEFTX"),
            left_y=self._axis("CONTROLLER_AXIS_LEFTY"),
            right_x=self._axis("CONTROLLER_AXIS_RIGHTX"),
            right_y=self._axis("CONTROLLER_AXIS_RIGHTY"),
            left_trigger=self._axis("CONTROLLER_AXIS_TRIGGERLEFT", trigger=True),
            right_trigger=self._axis("CONTROLLER_AXIS_TRIGGERRIGHT", trigger=True),
            a=self._button("CONTROLLER_BUTTON_A"),
            b=self._button("CONTROLLER_BUTTON_B"),
            x=self._button("CONTROLLER_BUTTON_X"),
            y=self._button("CONTROLLER_BUTTON_Y"),
            lb=self._button("CONTROLLER_BUTTON_LEFTSHOULDER"),
            rb=self._button("CONTROLLER_BUTTON_RIGHTSHOULDER"),
            view=self._button("CONTROLLER_BUTTON_BACK"),
            menu=self._button("CONTROLLER_BUTTON_START"),
            left_stick=self._button("CONTROLLER_BUTTON_LEFTSTICK"),
            right_stick=self._button("CONTROLLER_BUTTON_RIGHTSTICK"),
            timestamp=time.monotonic(),
        )

    def disconnect(self) -> None:
        if self.controller is not None:
            try:
                self.controller.quit()
            except Exception:
                pass
            self.controller = None
        if self.controller_module is not None:
            try:
                self.controller_module.quit()
            except Exception:
                pass
        if self.pygame is not None:
            try:
                self.pygame.quit()
            except Exception:
                pass


class XboxLeRobotTeleop:
    """Small LeRobot-style teleoperator wrapper for the Xbox controller."""

    def __init__(
        self,
        *,
        controller_index: int = 0,
        deadzone: float = 0.10,
        trigger_threshold: float = 0.15,
        require_xbox_name: bool = True,
    ):
        self.backend = PygameXboxBackend(controller_index, require_xbox_name)
        self.mapper = XboxActionMapper(deadzone, trigger_threshold)
        self.state = XboxState()

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    @property
    def is_connected(self) -> bool:
        return self.backend.is_connected

    @property
    def name(self) -> str:
        return self.backend.name

    def connect(self) -> None:
        self.backend.connect()
        self.update()

    def update(self) -> XboxState:
        self.state = self.backend.read()
        return self.state

    def get_action(self, *, update: bool = True) -> dict[str, float | int]:
        if update:
            self.update()
        return self.mapper.map_action(self.state)

    def get_teleop_events(self, *, update: bool = False) -> dict[str, bool]:
        if update:
            self.update()
        return self.mapper.map_events(self.state)

    def disconnect(self) -> None:
        self.backend.disconnect()
