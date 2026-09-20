"""Export frame-aligned annotation videos from MTO episode HDF5 files."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def export_video(episode_path: Path, output_path: Path, camera: str, fps: float) -> None:
    with h5py.File(episode_path, "r") as episode:
        frames = episode[f"/observation/{camera}/rgb"]
        first_frame = cv2.imdecode(np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        writer.write(first_frame)
        for frame_index in range(1, len(frames)):
            frame = cv2.imdecode(
                np.frombuffer(frames[frame_index], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            writer.write(frame)
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True, help="Episode group containing data/.")
    parser.add_argument("--fps", type=float, required=True, help="Recorded frame rate of the source episodes.")
    parser.add_argument(
        "--camera", choices=["head_camera", "left_camera", "right_camera"], default="head_camera"
    )
    parser.add_argument("--instruction", help="Task instruction to write for episodes without an instruction file.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing videos.")
    args = parser.parse_args()

    video_dir = args.root_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir = args.root_dir / "instructions"
    if args.instruction is not None:
        instruction_dir.mkdir(parents=True, exist_ok=True)

    for episode_path in sorted((args.root_dir / "data").glob("episode*.hdf5")):
        instruction_path = instruction_dir / f"{episode_path.stem}.json"
        if args.instruction is not None and not instruction_path.exists():
            instruction_path.write_text(
                json.dumps({"seen": [args.instruction]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        output_path = video_dir / f"{episode_path.stem}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"Keeping existing video: {output_path}")
            continue
        export_video(episode_path, output_path, args.camera, args.fps)
        print(f"Wrote video: {output_path}")


if __name__ == "__main__":
    main()
