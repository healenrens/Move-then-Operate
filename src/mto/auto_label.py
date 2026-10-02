"""Annotate standalone robot videos or LeRobot v3 episodes with phase labels."""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import cv2
from openai import APIConnectionError, APIStatusError, OpenAI
import numpy as np
from PIL import Image, ImageDraw

from mto.labels import label_metadata_path, label_path, load_phase_labels, validate_phase_labels

if TYPE_CHECKING:
    from mto.lerobot_reader import EpisodeRef, LeRobotReader
    from mto.dataset import Hdf5Episode, Hdf5Reader


RULE_VERSION = "move_operate_alignment_v6"
CAMERA_KEYS = ("observation.images.cam_high", "observation.images.cam_left_wrist",
               "observation.images.cam_right_wrist")
PHASE_RULES = """MOVE means coarse travel, ordinary carrying, and withdrawal.
OPERATE includes target-relative fine alignment before grasp, the final controlled
approach, the entire grasp/release action, and object repositioning. Start OPERATE
at the first visible fine alignment, not only at the gripper closing frame.
An empty gripper opening or closing in preparation is not evidence of object
contact. Combine the visual target relation with measured gripper changes.
A measured gripper event is motion evidence, not a contact sensor.
"""


@dataclass
class VideoSample:
    video_path: str
    instruction_path: str
    output_path: str


@dataclass
class AnnotationJob:
    output_path: Path
    metadata: dict
    frame_indices: list[int]
    video_path: str | None = None
    episode: EpisodeRef | Hdf5Episode | None = None

    @property
    def identity(self) -> str:
        return f"{self.metadata['relative_id']}/episode{self.metadata['episode_index']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_format", choices=("auto", "hdf5", "lerobot", "lerobot_v3"), default="hdf5")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--labels_root", help="External label root; mirrors <task>/<split>/auto_labels_v2.")
    parser.add_argument("--output_dir", help="Legacy standalone-video output directory; use labels_root for LeRobot.")
    parser.add_argument("--task_names", nargs="*", default=())
    parser.add_argument("--split_names", nargs="*", default=())
    parser.add_argument("--camera_key", help="Legacy standalone-video camera label. Structured episodes use head/left/right cameras.")
    parser.add_argument("--model_id", default=os.environ.get("ARK_MODEL_ID"))
    parser.add_argument("--api_key", help="Defaults to the ARK_API_KEY environment variable.")
    parser.add_argument("--base_url", default=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
    parser.add_argument("--sample_fps", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--gripper_event_threshold", type=float, default=0.02,
                        help="Minimum absolute per-frame gripper change in recorded units.")
    parser.add_argument("--event_context_frames", type=int, default=2)
    parser.add_argument("--mosaic_tile_size", type=int, default=320)
    parser.add_argument("--videos_limit", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Responses max_output_tokens, including thinking and final label JSON.")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max_attempts", type=int, default=3, help="Total API attempts per episode, including malformed responses.")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if not args.model_id:
        parser.error("Set ARK_MODEL_ID or pass --model_id.")
    if args.max_frames < 2 or args.sample_fps <= 0:
        parser.error("max_frames must be at least 2 and sample_fps must be positive.")
    if args.max_attempts < 1 or args.concurrency < 1:
        parser.error("max_attempts and concurrency must be positive.")
    if args.output_dir and args.labels_root:
        parser.error("Use only one of output_dir and labels_root.")
    if args.data_format == "auto":
        args.data_format = "lerobot" if next(Path(args.root_dir).glob("**/meta/info.json"), None) else "hdf5"
    if args.data_format == "lerobot_v3":
        args.data_format = "lerobot"
    if args.camera_key is None:
        args.camera_key = "observation.images.cam_high" if args.data_format == "lerobot" else "head_camera"
    if args.data_format == "lerobot" and args.output_dir:
        parser.error("LeRobot uses labels_root, not output_dir.")
    return args


def discover_dataset_pairs(root_dir: str, output_dir: str) -> List[VideoSample]:
    root = Path(root_dir)
    instruction_dir = root / "instructions"
    if not instruction_dir.is_dir():
        instruction_dir = root / "instruction"
    return [
        VideoSample(str(video), str(instruction_dir / f"{video.stem}.json"),
                    str(Path(output_dir) / f"{video.stem}_phases_labels_thinking.json"))
        for video in sorted((root / "video").glob("*.mp4"))
    ]


def read_instruction_text(instruction_path: str) -> str:
    with open(instruction_path, encoding="utf-8") as stream:
        data = json.load(stream)
    candidates = data.get("seen", data.get("instruction", [])) if isinstance(data, dict) else data
    if isinstance(candidates, str):
        return candidates.strip()
    # Stable annotation text makes the sidecar identity reproducible on resume.
    return next(str(value).strip() for value in candidates if isinstance(value, (str, int, float)) and str(value).strip())


def get_video_info(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0.0:
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Invalid frame count ({total_frames}) for video: {video_path}")
    return fps, total_frames


def frame_to_pil(frame) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def read_frame_at(video_path: str, frame_idx: int) -> Tuple[int, Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame_idx, frame_to_pil(frame)


def sample_frame_indices(length: int, fps: float, sample_fps: float, max_frames: int) -> list[int]:
    """Sample uniformly across the full episode, including both endpoints.

    sample_fps controls the desired density before the frame budget is applied.
    Long episodes retain full temporal coverage with a lower effective density.
    """
    if length == 1:
        return [0]
    desired_count = math.ceil((length - 1) * sample_fps / fps) + 1
    count = min(length, max_frames, max(2, desired_count))
    return [round(position * (length - 1) / (count - 1)) for position in range(count)]


def extract_gripper_events(state: np.ndarray, fps: float, threshold: float) -> list[dict]:
    """Group consecutive measured gripper changes; do not infer object contact."""
    events = []
    for arm, column in (("left", 6), ("right", 13)):
        values = state[:, column]
        delta = np.diff(values)
        active = np.flatnonzero(np.abs(delta) >= threshold)
        breaks = (np.diff(active) != 1) | (np.sign(delta[active[1:]]) != np.sign(delta[active[:-1]]))
        groups = np.split(active, np.flatnonzero(breaks) + 1)
        for group in groups:
            if not len(group):
                continue
            start, end = int(group[0]), int(group[-1]) + 1
            events.append({"arm": arm, "start_frame_idx": start, "end_frame_idx": end,
                           "start_seconds": start / fps, "end_seconds": end / fps,
                           "before": float(values[start]), "after": float(values[end]),
                           "motion": "increasing" if values[end] > values[start] else "decreasing"})
    return sorted(events, key=lambda event: (event["start_frame_idx"], event["arm"]))


def sample_event_frames(length, fps, sample_fps, max_frames, events, context_frames):
    """Prioritize endpoints and event anchors within a fixed full-episode budget."""
    selected = {0, length - 1}
    anchors = sorted({frame for event in events
                      for frame in (event["start_frame_idx"], event["end_frame_idx"])
                      if frame not in selected})
    room = max_frames - len(selected)
    if len(anchors) > room:
        anchors = [anchors[index] for index in np.linspace(0, len(anchors) - 1, room, dtype=int)] if room else []
    selected.update(anchors)
    context = sorted({max(0, min(length - 1, frame + offset)) for frame in anchors
                      for offset in (-context_frames, context_frames)} - selected)
    room = max_frames - len(selected)
    if len(context) > room:
        context = [context[index] for index in np.linspace(0, len(context) - 1, room, dtype=int)] if room else []
    selected.update(context)
    uniform = [frame for frame in sample_frame_indices(length, fps, sample_fps, max_frames)
               if frame not in selected]
    room = max_frames - len(selected)
    if len(uniform) > room:
        uniform = [uniform[index] for index in np.linspace(0, len(uniform) - 1, room, dtype=int)] if room else []
    return sorted(selected.union(uniform))


def make_camera_mosaic(images: list[Image.Image], camera_keys, tile_size: int) -> Image.Image:
    """Lay out synchronized head, left wrist, and right wrist observations."""
    mosaic = Image.new("RGB", (tile_size * len(images), tile_size + 24))
    draw = ImageDraw.Draw(mosaic)
    for index, (image, key) in enumerate(zip(images, camera_keys, strict=True)):
        mosaic.paste(image.convert("RGB").resize((tile_size, tile_size)), (index * tile_size, 24))
        draw.text((index * tile_size + 4, 4), key.removeprefix("observation.images."), fill="white")
    return mosaic


def build_calibration_messages(instruction, total_frames, samples, fps, draft, evidence, visual_layout=None):
    messages = build_messages(instruction, total_frames, samples, fps, visual_layout)
    messages[-1]["content"].append({"type": "input_text", "text": (
        "Boundary calibration stage. Use the measured gripper timeline and the synchronized images "
        "to correct guessed temporal boundaries, including the start of fine alignment. "
        "Measured changes may represent empty-gripper preparation; do not equate them with contact. "
        "Preserve exactly the draft's subtask count and order, phase counts, phase types/order, "
        "primary arms, descriptions, object names, and coordinates. Change only start_frame_idx "
        "and end_frame_idx, keeping full inclusive coverage. Return the complete calibrated JSON array."
        + "\nMeasured temporal evidence: " + json.dumps(evidence)
        + "\nVisual draft: " + json.dumps(draft))})
    return messages


def calibration_error(draft, calibrated) -> str:
    """Calibration can move boundaries, but cannot rewrite the semantic draft."""
    def semantics(labels):
        return [{key: ([{k: v for k, v in phase.items() if k not in ("start_frame_idx", "end_frame_idx")}
                       for phase in value] if key == "phases" else value)
                 for key, value in subtask.items() if key not in ("start_frame_idx", "end_frame_idx")}
                for subtask in labels]
    return "Calibration changed the draft's structure or semantics." if semantics(draft) != semantics(calibrated) else ""


def redact_error(text: str, client) -> str:
    keys = [getattr(client, "api_key", None)]
    keys.extend(os.environ.get(name) for name in ("ARK_API_KEY", "WANDB_API_KEY", "OPENAI_API_KEY"))
    for key in sorted({key for key in keys if isinstance(key, str) and key}, key=len, reverse=True):
        text = text.replace(key, "[REDACTED]")
    return text


def sample_video_frames(video_path: str, desired_sample_fps: float, max_frames: int) -> List[Tuple[int, Image.Image]]:
    fps, length = get_video_info(video_path)
    return [read_frame_at(video_path, index) for index in sample_frame_indices(length, fps, desired_sample_fps, max_frames)]


def to_data_url(image: Image.Image) -> str:
    from io import BytesIO

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_messages(
    instruction: str,
    total_frames: int,
    samples: List[Tuple[int, Image.Image]],
    fps: float,
    visual_layout: dict | None = None,
) -> List[Dict[str, object]]:
    frame_list_text = ", ".join(str(idx) for idx, _ in samples)
    prompt = (
        "You are an expert robotic manipulation annotator.\n"
        + f"Annotation rule: {RULE_VERSION}.\n" + PHASE_RULES + "\n"
        "Instruction: "
        + instruction
        + "\n"
        + f"Video cadence: {fps:g} FPS. Total frames: {total_frames}."
        + "\nYour task (follow strictly):\n"
        "1) Segment the entire video into consecutive subtasks that fully cover [0, total_frames-1] without gaps/overlaps.\n"
        "2) For each subtask, output phases (temporal slices within the subtask):\n"
        "   - phase_type in {move, operate}\n"
        "   - A subtask may have ONE phase (move OR operate) OR TWO phases (exactly one move and one operate).\n"
        "   - Do NOT repeat the same phase_type within a subtask. If you observe another movement/operation after the first occurence, start a NEW subtask for the additional action instead of duplicating the phase inside the same subtask.\n"
        "   - Order phases according to actual motion. It's allowed that operate appears before move if that matches the video.\n"
        "3) Identify the primary_arm (left/right/both/unknown) and give a concise English subtask_description.\n"
        "4) Predict normalized coordinates (top-right origin (0,0), bottom-left (1,1)) for: target_object_axis, left_gripper_end_axis, right_gripper_end_axis. Use [-1,-1] if a gripper is absent.\n"
        "5) Label only the motion actually present. An entire episode may contain only operate phases; never invent a move phase.\n"
        "6) Output STRICTLY a JSON array. No extra text, no markdown fences.\n"
        "Schema:\n"
        "[\n"
        "  {\n"
        '    "subtask": 1,\n'
        '    "subtask_description": "concise description in English",\n'
        '    "primary_arm": "left/right/both/unknown",\n'
        '    "start_frame_idx": 0,\n'
        '    "end_frame_idx": 120,\n'
        '    "target_object_name": "object name",\n'
        '    "target_object_axis": [0.42, 0.53],\n'
        '    "left_gripper_end_axis": [-1.0, -1.0],\n'
        '    "right_gripper_end_axis": [0.21, 0.78],\n'
        '    "phases": [\n'
        '      {\n'
        '        "phase_type": "move",\n'
        '        "phase_description": "short intent for the movement",\n'
        '        "start_frame_idx": 0,\n'
        '        "end_frame_idx": 45\n'
        "      },\n"
        "      {\n"
        '        "phase_type": "operate",\n'
        '        "phase_description": "short intent for the manipulation",\n'
        '        "start_frame_idx": 46,\n'
        '        "end_frame_idx": 120\n'
        "      }\n"
        "    ]\n"
        "  }\n"
        "]\n"
        "Rules:\n"
        "- Subtasks sorted by subtask index.\n"
        "- Frame indices are integers within [0, total_frames-1].\n"
        "- Phases are consecutive within a subtask: first phase starts at subtask start; last phase ends at subtask end.\n"
        "- Phase types limited to 'move'/'operate'. Within a subtask: one phase (move or operate) OR exactly two phases (one move and one operate). No duplicates; split into a new subtask when needed.\n"
        "- Endpoints are INCLUSIVE episode-local state-frame indices. The next interval starts at the previous end + 1.\n"
        "- Subtasks must start at 0 and finish at total_frames-1. Phases must fully cover their subtask without gaps or overlaps.\n"
        "- Single-frame phases are valid. Pure-operate episodes are valid.\n"
        "- Output JSON array only (no commentary, no markdown).\n"
    )

    content: List[Dict[str, object]] = [{"type": "input_text", "text": "Frame indices: " + frame_list_text}]
    content.append({"type": "input_text", "text": prompt})
    if visual_layout is not None:
        content.append({"type": "input_text", "text": (
            "Visual layout: " + json.dumps(visual_layout)
            + ". Each mosaic contains views at the SAME episode frame; tiles are ordered left to right "
              "as listed in cameras. Object/gripper coordinates refer to the first camera tile, "
              "normalized within that tile, excluding its header.")})
    for idx, image in samples:
        content.append({"type": "input_text", "text": f"frame_idx={idx}"})
        content.append(
            {
                "type": "input_image",
                "image_url": to_data_url(image),
            }
        )
    content.append({"type": "input_text", "text": prompt})
    return [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": "You output only valid JSON following the requested schema."}
            ],
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def sidecar_path(output_path: str, suffix: str) -> str:
    base, _ = os.path.splitext(output_path)
    return base + suffix


def compose_messages_with_feedback(
    base_messages: List[Dict[str, object]],
    feedbacks: List[str],
) -> List[Dict[str, object]]:
    messages = deepcopy(base_messages)
    if feedbacks:
        issues = " | ".join(feedbacks[-2:])
        guidance = (
            "Previous attempt issues: "
            + issues
            + ". Fix by ensuring: 1) Do not duplicate a phase_type inside a subtask; instead start a new subtask for the extra action. "
              "2) Each subtask has 1 phase (move/operate) or exactly 2 (one move + one operate) in real temporal order. "
              "3) Inclusive intervals cover the whole episode and each subtask exactly, with next start = previous end + 1. "
              "4) Do not invent a move phase in a pure-operate episode. "
              "Return JSON array only."
        )
        messages[-1]["content"].append({"type": "input_text", "text": guidance})
    return messages


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def run_inference(
    client: OpenAI,
    model_id: str,
    messages: List[Dict[str, object]],
    max_new_tokens: int,
) -> str:
    response = client.responses.create(
        model=model_id,
        input=messages,
        max_output_tokens=max_new_tokens,
        temperature=0.0,
    )
    # The SDK concatenates output_text message parts and excludes reasoning.
    return response.output_text


def extract_json_array(text: str) -> List[Dict[str, object]]:
    trimmed = text.strip()
    if trimmed.startswith("```"):
        first_break = trimmed.find("\n")
        last_ticks = trimmed.rfind("```")
        if first_break == -1 or last_ticks == -1:
            raise ValueError("Malformed fenced block.")
        trimmed = trimmed[first_break + 1 : last_ticks].strip()
    start = trimmed.find("[")
    end = trimmed.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON array not found in model output.")
    snippet = trimmed[start : end + 1]
    return json.loads(snippet)


def ensure_axis_pair(axis: object) -> Tuple[List[float], str]:
    if not isinstance(axis, list) or len(axis) != 2 or any(type(value) not in (int, float) for value in axis):
        return [], f"Axis must contain two numbers: {axis}"
    x, y = axis
    if x == -1 and y == -1:
        return [-1.0, -1.0], ""
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return [], f"Axis values must lie within [0,1] or be [-1,-1]: {axis}"
    return [round(float(x), 4), round(float(y), 4)], ""


def validate_subtasks(subtasks: object, total_frames: int) -> Tuple[List[Dict[str, object]], str]:
    """Validate the complete annotation schema and exact temporal coverage."""
    temporal_error = validate_phase_labels(subtasks, total_frames)
    if temporal_error:
        return [], temporal_error
    validated = []
    for index, subtask in enumerate(subtasks, start=1):
        if type(subtask.get("subtask")) is not int or subtask["subtask"] != index:
            return [], "Subtask indices must be integers sequentially starting at 1."
        for field in ("subtask_description", "target_object_name", "primary_arm"):
            if not isinstance(subtask.get(field), str) or not subtask[field].strip():
                return [], f"Subtask {index}: {field} must be a nonempty string."
        arm = subtask["primary_arm"].strip().lower()
        if arm not in ("left", "right", "both", "unknown"):
            return [], f"Subtask {index}: unsupported primary_arm {arm}."
        item = {key: subtask[key] for key in ("subtask", "start_frame_idx", "end_frame_idx")}
        item.update(primary_arm=arm, subtask_description=subtask["subtask_description"].strip(),
                    target_object_name=subtask["target_object_name"].strip())
        for field in ("target_object_axis", "left_gripper_end_axis", "right_gripper_end_axis"):
            axis, error = ensure_axis_pair(subtask.get(field))
            if error:
                return [], f"Subtask {index}: {field}: {error}"
            item[field] = axis
        phases = subtask["phases"]
        types = [phase["phase_type"] for phase in phases]
        if len(types) > 2 or len(set(types)) != len(types):
            return [], f"Subtask {index}: use at most one move and one operate phase; split repeated phases into another subtask."
        item["phases"] = []
        for phase in phases:
            if not isinstance(phase.get("phase_description"), str) or not phase["phase_description"].strip():
                return [], f"Subtask {index}: phase_description must be a nonempty string."
            item["phases"].append({
                "phase_type": phase["phase_type"],
                "phase_description": phase["phase_description"].strip(),
                "start_frame_idx": phase["start_frame_idx"],
                "end_frame_idx": phase["end_frame_idx"],
            })
        validated.append(item)
    return validated, ""


def resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_key = os.environ.get("ARK_API_KEY")
    if env_key:
        return env_key
    raise ValueError("API key required. Use --api_key or set ARK_API_KEY.")


def build_metadata(args: argparse.Namespace, relative_id: str, episode_index: int | str,
                   length: int, fps: float, instruction: str, source: dict,
                   events: list[dict] | None = None, cameras: tuple[str, ...] | None = None) -> dict:
    cameras = cameras if cameras is not None else (args.camera_key,)
    events = events if events is not None else []
    indices = sample_event_frames(length, fps, args.sample_fps, args.max_frames, events,
                                  args.event_context_frames)
    return {
        "rule_version": RULE_VERSION,
        "stages": ["visual_draft", "boundary_calibration"],
        "visual_layout": {"method": "same_frame_three_camera_mosaic" if len(cameras) == 3 else "single_camera",
                          "cameras": list(cameras), "tile_size": args.mosaic_tile_size},
        "temporal_evidence": {"method": "measured_gripper_changes" if source.get("data_format") in ("lerobot_v3", "hdf5") else "visual_only",
                              "gripper_event_threshold": args.gripper_event_threshold,
                              "event_context_frames": args.event_context_frames, "events": events},
        "relative_id": relative_id,
        "episode_index": episode_index,
        "length": length,
        "fps": fps,
        "camera": cameras[0],
        "instruction": instruction,
        "frame_space": "episode_local",
        "end_convention": "inclusive",
        "model": {"id": args.model_id, "base_url": args.base_url, "api": "responses",
                  "max_new_tokens": args.max_new_tokens},
        "sampling": {"method": "event_anchors_and_uniform_full_episode", "sample_fps": args.sample_fps,
                     "max_frames": args.max_frames, "frame_indices": indices},
        "source": source,
    }


def existing_label_error(job: AnnotationJob) -> str | None:
    labels, error = load_phase_labels(job.output_path, job.metadata["length"])
    if error:
        return error
    _, error = validate_subtasks(labels, job.metadata["length"])
    if error:
        return error
    metadata_path = label_metadata_path(job.output_path)
    try:
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return f"Cannot read metadata {metadata_path}: {error}"
    if not isinstance(metadata, dict):
        return "Label metadata must be an object."
    stage_attempts = metadata.get("stage_attempts", {})
    if any(type(stage_attempts.get(stage)) is not int or stage_attempts[stage] < 1
           for stage in ("visual_draft", "boundary_calibration")):
        return "Label metadata must record successful attempts for both stages."
    for key, expected in job.metadata.items():
        if metadata.get(key) != expected:
            return f"Label metadata mismatch for {key}."
    return None


def archive_invalid_labels(path: Path) -> list[str]:
    """Retain rejected annotations outside the filenames consumed by training."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived_label = path.with_name(f"{path.stem}.invalid.{timestamp}.json")
    path.rename(archived_label)
    archived = [str(archived_label)]
    metadata_path = label_metadata_path(path)
    if metadata_path.exists():
        archived_metadata = label_metadata_path(archived_label)
        metadata_path.rename(archived_metadata)
        archived.append(str(archived_metadata))
    return archived


def make_jobs(args: argparse.Namespace) -> tuple[list[AnnotationJob], LeRobotReader | Hdf5Reader | None, list[dict]]:
    root = Path(args.root_dir).expanduser().resolve()
    labels_root = Path(args.labels_root).expanduser().resolve() if args.labels_root else None
    jobs, failures = [], []
    if args.data_format == "lerobot":
        from mto.lerobot_reader import LeRobotReader

        reader = LeRobotReader(root, task_names=args.task_names, split_names=args.split_names)
        episodes = reader.episodes[:args.videos_limit] if args.videos_limit > 0 else reader.episodes
        for episode in episodes:
            output = label_path(episode.dataset_root, episode.relative_id, episode.episode_index, labels_root)
            instruction = reader.get_instruction(episode)
            source = {
                "data_format": "lerobot_v3", "dataset_root": str(episode.dataset_root),
                "codebase_version": episode.info["codebase_version"],
                "data_path": episode.info["data_path"], "video_path": episode.info["video_path"],
                "episode_metadata": {
                    key: value for key, value in episode.metadata.items()
                    if key in ("dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index")
                    or any(key.startswith(f"videos/{camera}/") for camera in CAMERA_KEYS)
                },
            }
            events = extract_gripper_events(reader.read_rows(episode)["observation.state"],
                                            episode.fps, args.gripper_event_threshold)
            metadata = build_metadata(args, episode.relative_id, episode.episode_index,
                                      episode.length, episode.fps, instruction, source, events, CAMERA_KEYS)
            jobs.append(AnnotationJob(output, metadata, metadata["sampling"]["frame_indices"], episode=episode))
        return jobs, reader, failures

    from mto.dataset import Hdf5Reader

    reader = Hdf5Reader(root)
    hdf5_episodes = {f"episode{ep.episode_index}": ep for ep in reader.episodes if ep.dataset_root == root}
    relative_id = f"{root.parent.name}/{root.name}"
    output_dir = Path(args.output_dir) if args.output_dir else (
        (labels_root / relative_id if labels_root else root) / "auto_labels_v2"
    )
    samples = discover_dataset_pairs(str(root), str(output_dir))
    if args.videos_limit > 0:
        samples = samples[:args.videos_limit]
    for sample in samples:
        try:
            instruction = read_instruction_text(sample.instruction_path)
            fps, length = get_video_info(sample.video_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, StopIteration, RuntimeError) as error:
            failures.append({"source": sample.video_path, "error": str(error)})
            continue
        video = Path(sample.video_path)
        episode_id = video.stem.removeprefix("episode")
        episode_index = int(episode_id) if episode_id.isdecimal() else video.stem
        stat = video.stat()
        source = {"data_format": "hdf5_video", "dataset_root": str(root), "video_path": str(video),
                  "instruction_path": sample.instruction_path, "video_size": stat.st_size,
                  "video_mtime_ns": stat.st_mtime_ns}
        episode = hdf5_episodes.get(video.stem)
        events, cameras = [], (args.camera_key,)
        if episode is not None:
            events = extract_gripper_events(reader.read_rows(episode)["observation.state"], fps,
                                            args.gripper_event_threshold)
            cameras = CAMERA_KEYS
            source.update(data_format="hdf5", hdf5_path=str(episode.path),
                          hdf5_mtime_ns=episode.path.stat().st_mtime_ns)
        metadata = build_metadata(args, relative_id, episode_index, length, fps, instruction, source, events, cameras)
        jobs.append(AnnotationJob(Path(sample.output_path), metadata, metadata["sampling"]["frame_indices"],
                                  video_path=sample.video_path, episode=episode))
    return jobs, reader, failures


def process_job(client: OpenAI, args: argparse.Namespace, job: AnnotationJob,
                reader: LeRobotReader | Hdf5Reader | None) -> dict:
    if job.episode is not None:
        cameras = job.metadata["visual_layout"]["cameras"]
        views = [reader.read_rgb(job.episode, camera, job.frame_indices) for camera in cameras]
        samples = [(index, make_camera_mosaic(list(images), cameras, args.mosaic_tile_size))
                   for index, images in zip(job.frame_indices, zip(*views, strict=True), strict=True)]
    else:
        samples = [read_frame_at(job.video_path, index) for index in job.frame_indices]

    def run_stage(name, base_messages, draft=None):
        feedbacks, final_raw = [], ""
        for attempt in range(1, args.max_attempts + 1):
            messages = compose_messages_with_feedback(base_messages, feedbacks)
            try:
                final_raw = redact_error(run_inference(client, args.model_id, messages, args.max_new_tokens), client)
                parsed = extract_json_array(final_raw)
                validated, error = validate_subtasks(parsed, job.metadata["length"])
                if not error and draft is not None:
                    error = calibration_error(draft, validated)
            except (APIConnectionError, APIStatusError, json.JSONDecodeError, ValueError, KeyError, TypeError, IndexError) as exception:
                error = redact_error(f"{type(exception).__name__}: {exception}", client)
            if not error:
                return validated, attempt, feedbacks, final_raw
            feedbacks.append(error)
            print(f"[retry] {job.identity} {name} attempt {attempt}/{args.max_attempts}: {error}")
        return [], attempt, feedbacks, final_raw

    draft_messages = build_messages(job.metadata["instruction"], job.metadata["length"], samples,
                                    job.metadata["fps"], job.metadata["visual_layout"])
    draft, draft_attempts, errors, draft_raw = run_stage("visual_draft", draft_messages)
    validated, calibration_attempts, calibration_raw = [], 0, ""
    if draft:
        calibration_messages = build_calibration_messages(
            job.metadata["instruction"], job.metadata["length"], samples, job.metadata["fps"], draft,
            job.metadata["temporal_evidence"], job.metadata["visual_layout"])
        validated, calibration_attempts, calibration_errors, calibration_raw = run_stage(
            "boundary_calibration", calibration_messages, draft)
        errors += calibration_errors
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path(str(job.output_path), "_phases_raw.txt")).write_text(
        "Visual draft:\n" + draft_raw + "\nBoundary calibration:\n" + calibration_raw, encoding="utf-8")
    error_path = Path(sidecar_path(str(job.output_path), "_phases_error.txt"))
    if not validated:
        error_path.write_text("\n".join(errors), encoding="utf-8")
        return {"identity": job.identity, "path": str(job.output_path), "status": "failed", "errors": errors}
    job.output_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {**job.metadata, "created_at": datetime.now(timezone.utc).isoformat(),
                "attempts": draft_attempts + calibration_attempts,
                "stage_attempts": {"visual_draft": draft_attempts, "boundary_calibration": calibration_attempts}}
    label_metadata_path(job.output_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if error_path.exists():
        error_path.unlink()
    print(f"Saved labels: {job.output_path}")
    return {"identity": job.identity, "path": str(job.output_path), "status": "annotated"}


def main() -> int:
    args = parse_args()
    jobs, reader, source_errors = make_jobs(args)
    summary = {"selected": len(jobs) + len(source_errors), "valid_existing": 0,
               "missing_labels_before": 0, "invalid_labels_before": 0, "annotated": 0,
               "failed": len(source_errors), "source_errors": source_errors, "invalid_existing": [], "failures": []}
    pending = []
    for job in jobs:
        if not job.output_path.is_file():
            summary["missing_labels_before"] += 1
        else:
            error = existing_label_error(job)
            if error:
                summary["invalid_labels_before"] += 1
                archived = archive_invalid_labels(job.output_path)
                summary["invalid_existing"].append({"identity": job.identity, "path": str(job.output_path),
                                                    "error": error, "archived": archived})
            else:
                summary["valid_existing"] += 1
                if args.resume:
                    print(f"[resume] Valid labels and metadata: {job.identity}")
                    continue
        pending.append(job)
    if pending:
        api_key = resolve_api_key(args.api_key)

        def run_single(job: AnnotationJob) -> dict:
            local_reader = copy(reader) if reader is not None else None
            if local_reader is not None:
                local_reader._row_cache = OrderedDict()
            with create_client(api_key, args.base_url) as client:
                return process_job(client, args, job, local_reader)

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_single, job) for job in pending]
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "annotated":
                    summary["annotated"] += 1
                else:
                    summary["failed"] += 1
                    summary["failures"].append(result)
    summary["complete"] = summary["selected"] > 0 and summary["failed"] == 0
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
