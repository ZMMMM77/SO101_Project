"""Record the SO-101 training topics from the existing ROS 2 graph."""

import argparse
import os
from pathlib import Path

from .interfaces import TOPIC_TYPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New rosbag directory for this episode")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Recording already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.execvp(
        "ros2",
        ["ros2", "bag", "record", "--storage", "mcap", "--output", str(output),
         "--topics", *TOPIC_TYPES],
    )


if __name__ == "__main__":
    main()
