import argparse
import base64
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
from PIL import Image
from openai import OpenAI


@dataclass
class VideoSample:
    video_path: str
    instruction_path: str
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto label robot videos via Doubao vision model with phase-aware subtasks."
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <root_dir>/auto_labels_v2.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="doubao-seed-1-6-thinking-250715",
        #doubao-seed-1-6-251015
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Falls back to environment variable ARK_API_KEY when omitted.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://ark.cn-beijing.volces.com/api/v3",
    )
    parser.add_argument("--sample_fps", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--videos_limit", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of videos to label in parallel.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Maximum inference retries when validation fails.",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Disable resume behaviour (process all videos regardless of existing outputs).",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def discover_dataset_pairs(root_dir: str, output_dir: str) -> List[VideoSample]:
    video_dir = os.path.join(root_dir, "video")
    instruction_candidates = [
        os.path.join(root_dir, "instructions"),
        os.path.join(root_dir, "instruction"),
    ]
    instruction_dir = None
    for cand in instruction_candidates:
        if os.path.isdir(cand):
            instruction_dir = cand
            break
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Missing directory: {video_dir}")
    if instruction_dir is None:
        raise FileNotFoundError("Missing instruction directory.")
    os.makedirs(output_dir, exist_ok=True)

    samples: List[VideoSample] = []
    for name in sorted(os.listdir(video_dir)):
        if not name.lower().endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, name)
        base = os.path.splitext(name)[0]
        instruction_path = os.path.join(instruction_dir, base + ".json")
        if not os.path.isfile(instruction_path):
            raise FileNotFoundError(f"Missing instruction JSON: {instruction_path}")
        output_path = os.path.join(output_dir, base + "_phases_labels_thinking.json")
        samples.append(
            VideoSample(
                video_path=video_path,
                instruction_path=instruction_path,
                output_path=output_path,
            )
        )
    if not samples:
        raise FileNotFoundError("No .mp4 videos discovered.")
    return samples


def read_instruction_text(instruction_path: str) -> str:
    with open(instruction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "seen" in data and isinstance(data["seen"], list) and data["seen"]:
            candidates = [
                str(x).strip() for x in data["seen"] if isinstance(x, (str, int, float))
            ]
            if not candidates:
                raise ValueError(f"No usable candidates in 'seen': {instruction_path}")
            return random.choice(candidates)
        if "instruction" in data:
            return str(data["instruction"]).strip()
    if isinstance(data, list) and data:
        candidates = [str(x).strip() for x in data if isinstance(x, (str, int, float))]
        if candidates:
            return random.choice(candidates)
    raise ValueError(f"No usable instruction text in {instruction_path}")


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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


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


def sample_video_frames(
    video_path: str,
    desired_sample_fps: float,
    max_frames: int,
) -> List[Tuple[int, Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0.0:
        cap.release()
        raise ValueError(f"Invalid FPS ({video_fps}) for video: {video_path}")
    stride = max(1, int(round(video_fps / max(0.1, desired_sample_fps))))
    samples: List[Tuple[int, Image.Image]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            samples.append((frame_idx, frame_to_pil(frame)))
        frame_idx += 1
    cap.release()
    if not samples:
        raise RuntimeError(f"No frames sampled from video: {video_path}")
    if len(samples) > max_frames:
        step = max(1, len(samples) // max_frames)
        samples = samples[::step][:max_frames]
    collected = {idx for idx, _ in samples}
    if 0 not in collected:
        samples.insert(0, read_frame_at(video_path, 0))
    total_frames = int(frame_idx)
    last_frame_idx = total_frames - 1
    if last_frame_idx not in collected:
        samples.append(read_frame_at(video_path, last_frame_idx))
    samples.sort(key=lambda x: x[0])
    return samples


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
) -> List[Dict[str, object]]:
    frame_list_text = ", ".join(str(idx) for idx, _ in samples)
    prompt = (
        "You are an expert robotic manipulation annotator.\n"
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
        "5) Ensure the entire video contains AT LEAST ONE move phase among all subtasks.\n"
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
        "- The entire video must contain at least one move phase.\n"
        "- Output JSON array only (no commentary, no markdown).\n"
    )

    content: List[Dict[str, object]] = [{"type": "text", "text": "Frame indices: " + frame_list_text}]
    content.append({"type": "text", "text": prompt})
    for idx, image in samples:
        content.append({"type": "text", "text": f"frame_idx={idx}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": to_data_url(image),
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You output only valid JSON following the requested schema."}
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
              "3) The entire video contains at least one move phase. "
              "Return JSON array only."
        )
        messages[-1]["content"].append({"type": "text", "text": guidance})
    return messages


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def run_inference(
    client: OpenAI,
    model_id: str,
    messages: List[Dict[str, object]],
    max_new_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part["text"] for part in content if part.get("type") == "text"]
        if not text_parts:
            raise ValueError("Model returned empty content list.")
        return "".join(text_parts)
    raise ValueError("Unsupported response content type.")


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


def ensure_axis_pair(axis: List[float]) -> Tuple[List[float], str]:
    if len(axis) != 2:
        return [], f"Axis must have two elements: {axis}"
    x = float(axis[0])
    y = float(axis[1])
    if x == -1.0 and y == -1.0:
        return [-1.0, -1.0], ""
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return [], f"Axis values must lie within [0,1] or be [-1,-1]: {axis}"
    return [round(x, 4), round(y, 4)], ""


def normalize_primary_arm(value: str) -> Tuple[str, str]:
    lowered = value.strip().lower()
    if lowered in {"left", "right", "both", "unknown"}:
        return lowered, ""
    return "", f"Unsupported primary_arm value: {value}"


def validate_phases(
    phases: List[Dict[str, object]],
    subtask_start: int,
    subtask_end: int,
) -> Tuple[List[Dict[str, object]], str]:
    return _validate_phases_internal(phases, subtask_start, subtask_end, relaxed=False)


def validate_phases_relaxed(
    phases: List[Dict[str, object]],
    subtask_start: int,
    subtask_end: int,
) -> Tuple[List[Dict[str, object]], str]:
    return _validate_phases_internal(phases, subtask_start, subtask_end, relaxed=True)


def _validate_phases_internal(
    phases: List[Dict[str, object]],
    subtask_start: int,
    subtask_end: int,
    relaxed: bool,
) -> Tuple[List[Dict[str, object]], str]:
    if not phases:
        return [], "Each subtask must contain at least one phase."
    validated: List[Dict[str, object]] = []
    seen_types: set[str] = set()
    last_type: str | None = None
    for phase in phases:
        phase_type = str(phase["phase_type"]).strip().lower()
        if phase_type not in {"move", "operate"}:
            return [], f"Invalid phase_type: {phase_type}"
        if not relaxed:
            if phase_type in seen_types:
                return [], "Duplicate phase types within a single subtask are not allowed."
        else:
            if last_type is not None and phase_type == last_type:
                return [], "Consecutive duplicate phase types within a single subtask are not allowed."
        start_idx = int(phase["start_frame_idx"])
        end_idx = int(phase["end_frame_idx"])
        if end_idx < start_idx:
            return [], "Phase end must be >= start."
        if end_idx > subtask_end:
            return [], "Phase end exceeds subtask range."
        if start_idx < subtask_start:
            return [], f"Phase start {start_idx} is before subtask start {subtask_start}."
        if start_idx > subtask_end:
            return [], f"Phase start {start_idx} exceeds subtask end {subtask_end}."
        desc = str(phase["phase_description"]).strip()
        validated.append(
            {
                "phase_type": phase_type,
                "phase_description": desc,
                "start_frame_idx": start_idx,
                "end_frame_idx": end_idx,
            }
        )
        seen_types.add(phase_type)
        last_type = phase_type
    if not relaxed:
        if len(validated) == 2:
            phase_types = {validated[0]["phase_type"], validated[1]["phase_type"]}
            if phase_types != {"move", "operate"}:
                return [], "Two-phase subtasks must contain exactly one 'move' and one 'operate'."
        if len(validated) > 2:
            return [], "A subtask can contain at most two phases."
    else:
        if len(validated) > 5:
            return [], "Relaxed validation allows at most five phases per subtask."
        unique_types = {item["phase_type"] for item in validated}
        if len(unique_types) < 2:
            return [], "Relaxed validation still requires both move and operate when multiple phases are present."
    return validated, ""


def validate_subtasks(
    subtasks: List[Dict[str, object]],
    total_frames: int,
) -> Tuple[List[Dict[str, object]], str]:
    return _validate_subtasks_internal(subtasks, total_frames, relaxed=False)


def validate_subtasks_relaxed(
    subtasks: List[Dict[str, object]],
    total_frames: int,
) -> Tuple[List[Dict[str, object]], str]:
    return _validate_subtasks_internal(subtasks, total_frames, relaxed=True)


def _validate_subtasks_internal(
    subtasks: List[Dict[str, object]],
    total_frames: int,
    relaxed: bool,
) -> Tuple[List[Dict[str, object]], str]:
    if not isinstance(subtasks, list):
        return [], "Top-level JSON must be an array."
    if not subtasks:
        return [], "Model returned empty subtask list."
    validated: List[Dict[str, object]] = []
    has_move_phase = False
    for idx, subtask in enumerate(subtasks, start=1):
        subtask_id = int(subtask["subtask"])
        if subtask_id != idx:
            return [], "Subtask indices must be sequential starting at 1."
        start_idx = int(subtask["start_frame_idx"])
        end_idx = int(subtask["end_frame_idx"])
        if end_idx < start_idx:
            return [], f"Subtask {idx} end must be >= start."
        if end_idx >= total_frames:
            return [], f"Subtask {idx} end exceeds total frames."
        desc = str(subtask["subtask_description"]).strip()
        target_name = str(subtask["target_object_name"]).strip()
        primary_arm, arm_err = normalize_primary_arm(str(subtask["primary_arm"]))
        if arm_err:
            return [], f"Subtask {idx}: {arm_err}"
        axis_target, axis_err = ensure_axis_pair(list(subtask["target_object_axis"]))
        if axis_err:
            return [], f"Subtask {idx}: {axis_err}"
        axis_left, axis_left_err = ensure_axis_pair(list(subtask["left_gripper_end_axis"]))
        if axis_left_err:
            return [], f"Subtask {idx}: {axis_left_err}"
        axis_right, axis_right_err = ensure_axis_pair(list(subtask["right_gripper_end_axis"]))
        if axis_right_err:
            return [], f"Subtask {idx}: {axis_right_err}"
        phase_entries = list(subtask.get("phases", []))
        phase_validator = validate_phases_relaxed if relaxed else validate_phases
        phases, phases_err = phase_validator(
            phase_entries, start_idx, end_idx
        )
        if phases_err:
            return [], f"Subtask {idx}: {phases_err}"
        if any(phase["phase_type"] == "move" for phase in phases):
            has_move_phase = True
        validated.append(
            {
                "subtask": subtask_id,
                "subtask_description": desc,
                "primary_arm": primary_arm,
                "start_frame_idx": start_idx,
                "end_frame_idx": end_idx,
                "target_object_name": target_name,
                "target_object_axis": axis_target,
                "left_gripper_end_axis": axis_left,
                "right_gripper_end_axis": axis_right,
                "phases": phases,
            }
        )
    if validated[-1]["end_frame_idx"] != total_frames - 1:
        return [], "Last subtask must end at total_frames - 1."
    if not has_move_phase:
        return [], "At least one phase across the video must be of type 'move'."
    return validated, ""


def resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_key = os.environ.get("ARK_API_KEY")
    if env_key:
        return env_key
    raise ValueError("API key required. Use --api_key or set ARK_API_KEY.")


def process_video(
    client: OpenAI,
    model_id: str,
    sample: VideoSample,
    sample_fps: float,
    max_frames: int,
    max_new_tokens: int,
    max_attempts: int,
) -> None:
    instruction = read_instruction_text(sample.instruction_path)
    fps, total_frames = get_video_info(sample.video_path)
    samples = sample_video_frames(
        sample.video_path, desired_sample_fps=sample_fps, max_frames=max_frames
    )
    base_messages = build_messages(instruction, total_frames, samples, fps)
    feedbacks: List[str] = []
    final_validated: List[Dict[str, object]] = []
    final_raw = ""
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        messages = compose_messages_with_feedback(base_messages, feedbacks)
        raw_text = run_inference(
            client=client,
            model_id=model_id,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        final_raw = raw_text
        parsed = extract_json_array(raw_text)
        validated, validation_error = validate_subtasks(parsed, total_frames)
        if not validation_error:
            final_validated = validated
            break
        feedbacks.append(validation_error)
        print(f"[retry] {os.path.basename(sample.video_path)} attempt {attempt + 1}: {validation_error}")
    os.makedirs(os.path.dirname(sample.output_path), exist_ok=True)
    # Always save the last raw response for debugging, even if validation failed
    raw_path = sidecar_path(sample.output_path, "_phases_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(final_raw)
    if not final_validated:
        relaxed_validated, relaxed_error = validate_subtasks_relaxed(parsed, total_frames)
        if not relaxed_error:
            final_validated = relaxed_validated
            print(f"[warn] {os.path.basename(sample.video_path)} accepted via relaxed validation fallback.")
        else:
            err_path = sidecar_path(sample.output_path, "_phases_error.txt")
            last_error = relaxed_error or (feedbacks[-1] if feedbacks else "unknown validation failure")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(last_error)
            raise RuntimeError(f"Failed to obtain valid labels after {attempts} attempts: {last_error}")
    with open(sample.output_path, "w", encoding="utf-8") as f:
        json.dump(final_validated, f, ensure_ascii=False, indent=2)
    print(f"Saved labels: {sample.output_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.root_dir, "auto_labels_v2")
    samples = discover_dataset_pairs(args.root_dir, output_dir)
    if args.videos_limit > 0:
        samples = samples[: args.videos_limit]
    if args.resume:
        remaining: List[VideoSample] = []
        for sample in samples:
            if os.path.isfile(sample.output_path) and os.path.getsize(sample.output_path) > 0:
                print(f"[resume] Skipping existing: {os.path.basename(sample.output_path)}")
                continue
            remaining.append(sample)
        samples = remaining
    if not samples:
        raise RuntimeError("No videos to process after resume filtering.")
    api_key = resolve_api_key(args.api_key)

    def run_single(sample: VideoSample, position: int, total: int) -> None:
        print(f"[{position}/{total}] Processing {os.path.basename(sample.video_path)}")
        client_local = create_client(api_key=api_key, base_url=args.base_url)
        process_video(
            client=client_local,
            model_id=args.model_id,
            sample=sample,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            max_attempts=args.max_attempts,
        )

    total = len(samples)
    concurrency = max(1, int(args.concurrency))
    if concurrency == 1 or total == 1:
        for idx, sample in enumerate(samples, start=1):
            run_single(sample, idx, total)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(run_single, sample, idx, total): sample
                for idx, sample in enumerate(samples, start=1)
            }
            first_error: Exception | None = None
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    sample = futures[future]
                    print(f"[error] {os.path.basename(sample.video_path)} failed: {exc}")
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise first_error


if __name__ == "__main__":
    main()
