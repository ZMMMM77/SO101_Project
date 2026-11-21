"""Predict a SmolVLA action and send it to the SO-101 ROS follower."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from std_msgs.msg import String

from .interfaces import CAMERA_RENAME_MAP, CAMERAS, GRIPPER_POSITION, JOINT_STATES

POSITION_KEYS = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)


def load_policy(checkpoint, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.numpy import load_file

    definition = json.loads((checkpoint / "policy_postprocessor.json").read_text())
    step = next((step for step in definition["steps"]
                 if step.get("registry_name") == "unnormalizer_processor"), None)
    if not step or "state_file" not in step:
        raise ValueError("Checkpoint has no trained SO-101 action statistics")
    stats = load_file(checkpoint / step["state_file"])
    for key in ("action.mean", "action.std"):
        array = stats.get(key)
        if array is None or array.shape != (len(POSITION_KEYS),) or not np.isfinite(array).all():
            raise ValueError("Expected six-dimensional calibrated action statistics")
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg).eval()
    pre, post = make_pre_post_processors(
        cfg, checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": CAMERA_RENAME_MAP},
        },
    )
    return policy, pre, post


def recorded_frame(root, index):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/so101", root=root, video_backend="pyav")
    for key in ("observation.state", "action"):
        if dataset.meta.features[key].get("names") != list(POSITION_KEYS):
            raise ValueError(f"Unexpected calibrated units or joint order for {key}")
    if not 0 <= index < len(dataset):
        raise ValueError(f"Frame {index} is outside a dataset of {len(dataset)} frames")
    sample = dataset[index]
    return {key: sample[key] for key in (*CAMERA_RENAME_MAP, "observation.state", "task")}


def live_frame(node, task):
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64

    from .convert_mcap import joint_vector, rgb_image

    latest = {}
    types = {**dict.fromkeys(CAMERAS, Image), JOINT_STATES: JointState, GRIPPER_POSITION: Float64}

    def store(topic, message):
        received = node.get_clock().now().nanoseconds
        header = getattr(message, "header", None)
        stamp = received
        if header is not None:
            sent = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
            stamp = sent or received
        latest[topic] = (stamp, message)

    subscriptions = [
        node.create_subscription(kind, topic, lambda msg, t=topic: store(t, msg), qos_profile_sensor_data)
        for topic, kind in types.items()
    ]
    try:
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if len(latest) != len(types):
                continue
            stamp = min(value[0] for value in latest.values())
            if not 0 <= node.get_clock().now().nanoseconds - stamp <= 75_000_000:
                continue
            sample = {key: rgb_image(latest[t][1]) for t, key in CAMERAS.items()}
            sample["observation.state"] = joint_vector(latest[JOINT_STATES][1], latest[GRIPPER_POSITION][1])
            return prepare_observation_for_inference(sample, torch.device("cpu"), task, "so101_follower"), stamp
        raise RuntimeError("Could not obtain live state and both images within 75 ms")
    finally:
        for sub in subscriptions:
            node.destroy_subscription(sub)


def wait_until(node, check, timeout, description):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        if check():
            return
        rclpy.spin_once(node, timeout_sec=0.05)
    raise RuntimeError(description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Read one recorded frame and use the preview endpoint")
    source.add_argument("--live", action="store_true", help="Send one target from a fresh live observation")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for a recorded dataset")
    parser.add_argument("--task", help="Instruction override. Required for live inference.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.live and not (args.task and args.task.strip()):
        parser.error("--live requires --task")
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(project / ".hf-home"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(project / ".lerobot"))

    import torch

    mode = "live" if args.live else "preview"
    expected_stamp = None
    reply = {}

    def receive(message):
        try:
            result = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(result, dict):
            return
        if result.get("stamp_ns") == expected_stamp and result.get("mode") == mode:
            reply.update(result)

    rclpy.init()
    node = rclpy.create_node("so101_policy_inference")
    topic = "/so101/policy_action" if args.live else "/so101/policy_preview"
    publisher = node.create_publisher(String, topic, 1)
    subscription = node.create_subscription(String, "/so101/policy_result", receive, 10)
    try:
        wait_until(node, lambda: publisher.get_subscription_count() and subscription.get_publisher_count(),
                   6, f"No follower found for {topic}")
        model, preprocess, postprocess = load_policy(args.checkpoint.resolve(), args.device)
        if args.live:
            frame, expected_stamp = live_frame(node, args.task)
        else:
            frame = recorded_frame(args.dataset.resolve(), args.frame)
            if args.task:
                frame["task"] = args.task
        with torch.inference_mode():
            actions = postprocess(model.predict_action_chunk(preprocess(frame))).squeeze(0).detach().cpu().numpy()
        if actions.ndim != 2 or actions.shape[1] != len(POSITION_KEYS) or not len(actions):
            raise ValueError(f"Expected an action sequence of shape (steps, 6), received {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Policy generated nonfinite values")
        if not args.live:
            expected_stamp = node.get_clock().now().nanoseconds
        positions = dict(zip(POSITION_KEYS, actions[0].tolist(), strict=True))
        print(json.dumps({"predicted_steps": len(actions), "sent_step": 0, "positions": positions}), flush=True)
        publisher.publish(String(data=json.dumps({"stamp_ns": expected_stamp, "positions": positions})))
        wait_until(node, lambda: bool(reply), 6, "Follower did not acknowledge the target")
        print(json.dumps(reply), flush=True)
        if not reply["accepted"]:
            raise RuntimeError(reply["reason"])
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
