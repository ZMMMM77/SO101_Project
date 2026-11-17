"""Fine-tune the official SmolVLA base model on a local SO-101 dataset."""

import argparse
import json
import os
from pathlib import Path
import sys

from .interfaces import CAMERA_RENAME_MAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101")
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("Training steps and batch size must be positive")

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        parser.error(f"LeRobot dataset metadata was not found: {info_path}")
    if output.exists():
        parser.error(f"Training output already exists: {output}")
    with info_path.open() as info_file:
        features = json.load(info_file)["features"]
    for key in (*CAMERA_RENAME_MAP, "observation.state", "action"):
        if key not in features:
            parser.error(f"Dataset is missing feature {key}")

    project = Path(__file__).resolve().parents[1]
    source = project / "third_party" / "lerobot" / "src"
    if source.is_dir():
        existing_path = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(source) + (os.pathsep + existing_path if existing_path else "")
    os.environ.setdefault("HF_HOME", str(project / ".hf-home"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(project / ".lerobot"))

    os.execv(sys.executable, [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.root={dataset}", f"--dataset.repo_id={args.repo_id}",
        "--dataset.video_backend=pyav",
        f"--policy.path={args.policy_path}", f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--rename_map={json.dumps(CAMERA_RENAME_MAP)}",
        f"--output_dir={output}", f"--job_name={output.name}",
        f"--steps={args.steps}", f"--batch_size={args.batch_size}",
    ])


if __name__ == "__main__":
    main()
