"""Convert SO-101 ROS 2 MCAP episodes to a local LeRobotDataset v3 dataset."""

import argparse
from contextlib import closing
from itertools import chain, count
import math
from pathlib import Path
import tempfile

from .interfaces import CAMERAS, GRIPPER_POSITION, GRIPPER_TARGET
from .interfaces import JOINT_STATES, JOINT_TARGETS, TOPIC_TYPES


class TopicCursor:
    """Read one MCAP topic lazily and retain the samples around a query time."""

    def __init__(self, reader, message_type):
        from rclpy.serialization import deserialize_message

        self.reader = reader
        self.message_type = message_type
        self.deserialize = deserialize_message
        self.previous = None
        self.pending = self.read_next()

    def read_next(self):
        if not self.reader.has_next():
            return None
        _, data, stamp = self.reader.read_next()
        return stamp, self.deserialize(data, self.message_type)

    def before(self, stamp):
        while self.pending is not None and self.pending[0] <= stamp:
            self.previous = self.pending
            self.pending = self.read_next()
        return self.previous

    def nearest(self, stamp):
        previous = self.before(stamp)
        neighbors = (sample for sample in (previous, self.pending) if sample is not None)
        return min(neighbors, key=lambda sample: abs(sample[0] - stamp), default=None)


def open_topics(bag):
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    metadata = SequentialReader()
    metadata.open(StorageOptions(uri=str(bag), storage_id="mcap"), ConverterOptions("", ""))
    actual_types = {topic.name: topic.type for topic in metadata.get_all_topics_and_types()}
    invalid = [topic for topic, kind in TOPIC_TYPES.items() if actual_types.get(topic) != kind]
    if invalid:
        raise ValueError(f"{bag}: missing or incompatible ROS topics: {', '.join(invalid)}")
    del metadata

    tracks = {}
    for topic, kind in TOPIC_TYPES.items():
        reader = SequentialReader()
        reader.open(StorageOptions(uri=str(bag), storage_id="mcap"), ConverterOptions("", ""))
        reader.set_filter(StorageFilter(topics=[topic]))
        cursor = TopicCursor(reader, get_message(kind))
        if cursor.pending is None:
            raise ValueError(f"{bag}: {topic} contains no messages")
        tracks[topic] = cursor
    return tracks


def age_ok(sample, stamp, limit_ns):
    return sample is not None and abs(sample[0] - stamp) <= limit_ns


def joint_vector(joints, gripper):
    import numpy as np
    from so101_cartesian_control import ARM_JOINTS, model_rad_to_lerobot_deg

    values = dict(zip(joints.name, joints.position, strict=True))
    if len(values) != len(joints.name) or any(name not in values for name in ARM_JOINTS):
        raise ValueError("JointState must contain all five distinct SO-101 arm joints")
    positions = model_rad_to_lerobot_deg([values[name] for name in ARM_JOINTS])
    result = np.asarray([*positions, gripper.data], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("State or action contains non-finite values")
    return result


def rgb_image(message):
    import numpy as np

    if message.encoding not in {"bgr8", "rgb8"}:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    row_bytes = 3 * message.width
    if (
        message.width <= 0 or message.height <= 0 or message.step < row_bytes
        or len(message.data) != message.height * message.step
    ):
        raise ValueError("ROS image size, stride, and payload disagree")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :row_bytes].reshape(message.height, message.width, 3)
    return np.ascontiguousarray(image[..., ::-1] if message.encoding == "bgr8" else image)


def episode_frames(bag, task, fps, max_skew_ns):
    tracks = open_topics(bag)
    targets = tracks[JOINT_TARGETS]
    start_ns = targets.pending[0]
    tick_limit_ns = min(max_skew_ns, round(0.5e9 / fps))
    last_valid = None

    # Bag receive timestamps put stamped and headerless messages on one clock.
    for index in count():
        tick_ns = start_ns + round(index * 1e9 / fps)
        target = targets.nearest(tick_ns)
        if targets.pending is None and tick_ns > targets.previous[0]:
            break
        if not age_ok(target, tick_ns, tick_limit_ns):
            continue

        action_ns, joint_target = target
        gripper_target = tracks[GRIPPER_TARGET].nearest(action_ns)
        state = tracks[JOINT_STATES].before(action_ns)
        if not age_ok(gripper_target, action_ns, tick_limit_ns) or not age_ok(
            state, action_ns, max_skew_ns
        ):
            continue

        state_ns, joint_state = state
        gripper_state = tracks[GRIPPER_POSITION].nearest(state_ns)
        images = {key: tracks[topic].before(state_ns) for topic, key in CAMERAS.items()}
        if (
            not age_ok(gripper_state, state_ns, tick_limit_ns)
            or gripper_state[0] > action_ns
            or not all(age_ok(sample, state_ns, max_skew_ns) for sample in images.values())
        ):
            continue

        if last_valid is not None and index != last_valid + 1:
            raise ValueError(f"{bag}: missing synchronized data near {index / fps:.2f}s")
        last_valid = index
        yield {
            **{key: rgb_image(sample[1]) for key, sample in images.items()},
            "observation.state": joint_vector(joint_state, gripper_state[1]),
            "action": joint_vector(joint_target, gripper_target[1]),
            "task": task,
        }


def write_dataset(bags, output, repo_id, task, fps, max_skew_ns):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from so101_cartesian_control import ARM_OBSERVATION_KEYS, GRIPPER_KEY

    names = [*ARM_OBSERVATION_KEYS, GRIPPER_KEY]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".so101-convert-", dir=output.parent) as staging:
        root = Path(staging) / "dataset"
        dataset = None
        image_shapes = None
        for bag in bags:
            frames = episode_frames(bag, task, fps, max_skew_ns)
            with closing(frames):
                first = next(frames, None)
                if first is None:
                    raise ValueError(f"{bag}: no complete synchronized frames")
                if dataset is None:
                    image_shapes = {key: first[key].shape for key in CAMERAS.values()}
                    features = {
                        key: {"dtype": "video", "shape": shape,
                              "names": ["height", "width", "channels"]}
                        for key, shape in image_shapes.items()
                    }
                    features.update({
                        key: {"dtype": "float32", "shape": (len(names),), "names": names}
                        for key in ("observation.state", "action")
                    })
                    dataset = LeRobotDataset.create(
                        repo_id=repo_id, fps=fps, features=features, root=root,
                        robot_type="so101_follower", vcodec="h264", video_backend="pyav",
                    )
                total = 0
                for frame in chain((first,), frames):
                    if any(frame[key].shape != shape for key, shape in image_shapes.items()):
                        raise ValueError(f"{bag}: camera size differs from the dataset schema")
                    dataset.add_frame(frame)
                    total += 1
                if total < 2:
                    raise ValueError(f"{bag}: fewer than two synchronized frames")
                dataset.save_episode()
                print(f"{bag.name}: {total} frames at {fps} Hz", flush=True)

        dataset.finalize()
        if output.exists():
            raise FileExistsError(f"Dataset path already exists: {output}")
        root.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", type=Path, nargs="+", help="One MCAP or rosbag directory per episode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Instruction shared by these episodes")
    parser.add_argument("--repo-id", default="local/so101")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-skew-ms", type=float, default=75)
    args = parser.parse_args()
    if args.fps < 1 or not math.isfinite(args.max_skew_ms) or args.max_skew_ms <= 0:
        parser.error("FPS and maximum timestamp skew must be positive")
    if not args.task.strip():
        parser.error("The task instruction must contain text")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Dataset path already exists: {output}")
    bags = [bag.expanduser().resolve() for bag in args.bags]
    if len(set(bags)) != len(bags):
        parser.error("Specify each bag once")
    for bag in bags:
        if not (bag.is_dir() or (bag.suffix == ".mcap" and bag.is_file())):
            parser.error(f"Expected a rosbag directory or MCAP file: {bag}")
    try:
        write_dataset(bags, output, args.repo_id, args.task.strip(), args.fps,
                      round(args.max_skew_ms * 1e6))
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    print(f"Dataset saved to {output}")


if __name__ == "__main__":
    main()
