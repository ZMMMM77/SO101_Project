"""Validate SmolVLA actions before delivering them to the follower."""

import json
import math

from std_msgs.msg import String


class PolicyInput:
    def __init__(self, node, cart, args, mode, timeout):
        self.node = node
        self.cart = cart
        self.mode = mode
        self.timeout_ns = round(timeout * 1e9)
        self.pending = None
        self.latest_stamp = 0
        self.keys = (*cart.ARM_OBSERVATION_KEYS, cart.GRIPPER_KEY)
        self.kinematics = cart.SO101Kinematics(args.urdf, args.calibration, args.joint_margin_deg)
        self.results = node.create_publisher(String, "policy_result", 10)
        topic = "policy_preview" if mode == "preview" else "policy_action"
        self.subscription = node.create_subscription(String, topic, self.receive, 1)

    def receive(self, message):
        stamp = None
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            stamp = request.get("stamp_ns")
            positions = request.get("positions")
            if type(stamp) is not int or stamp <= 0:
                raise ValueError("Expected a positive integer stamp_ns")
            if not isinstance(positions, dict) or set(positions) != set(self.keys):
                raise ValueError("Expected the five calibrated joint keys and gripper.pos")
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in positions.values()):
                raise ValueError("All positions must be finite numbers")
            q = self.cart.lerobot_deg_to_model_rad([positions[key] for key in self.keys[:-1]])
            grip = float(positions[self.keys[-1]])
            if not 0.0 <= grip <= 100.0:
                raise ValueError("gripper.pos must be in the calibrated 0–100 range")
            violations = self.kinematics.joint_limit_violations(q, tolerance_deg=0.0)
            if violations:
                raise ValueError("; ".join(violations))
            self.kinematics.check_workspace(self.kinematics.fk(q).xyz_m)
            if self.mode == "live":
                if stamp <= self.latest_stamp or not self.fresh(stamp):
                    raise ValueError("Stale, future-dated or out-of-order observation")
                self.latest_stamp = stamp
                self.pending = (stamp, q, grip)
        except (ValueError, TypeError, OverflowError, self.cart.CartesianControlError) as error:
            self.reply(stamp, accepted=False, reason=str(error))
            return
        self.reply(
            stamp, accepted=True, positions=positions,
            reason="validated without motor writes" if self.mode == "preview" else "queued for the follower",
        )

    def fresh(self, stamp):
        return 0 <= self.node.get_clock().now().nanoseconds - stamp <= self.timeout_ns

    def target(self, enabled):
        if not enabled or (self.pending is not None and not self.fresh(self.pending[0])):
            self.pending = None
        return self.pending[1:] if self.pending is not None else None

    def reply(self, stamp, **result):
        self.results.publish(String(data=json.dumps({"stamp_ns": stamp, "mode": self.mode, **result})))
