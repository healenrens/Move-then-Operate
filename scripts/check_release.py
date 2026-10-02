#!/usr/bin/env python3
"""Check publication files without printing matched deployment or credential values."""
import argparse
from pathlib import Path
import re
import sys

RUNTIME_DIRS = {".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache", "wandb",
                "assets", "checkpoints", "data", "datasets", "logs", "outputs", "artifacts", "runtime",
                "eval_results", "mto_traces", "auto_labels_v2", "dist", "build"}
RUNTIME_SUFFIXES = {".hdf5", ".h5", ".mp4", ".npz", ".npy", ".log", ".safetensors", ".pt", ".pth", ".ckpt"}
PATTERNS = {
    "non-English text": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]"),
    "private path": re.compile(r"/(?:Users/[^/\s]+|root/(?!\.cache/)[^/\s]+|mnt/pfs/[^\s\"']+)"),
    "private address": re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"),
    "personal model endpoint": re.compile(r"\bep-\d{8,}-[A-Za-z0-9]+\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential candidate": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})\b"),
    "literal API key": re.compile(r'''(?:api_key|ARK_API_KEY|WANDB_API_KEY|OPENAI_API_KEY)\s*[:=]\s*["']([^"'\n]+)["']''', re.I),
}


def check_files(root: Path, paths) -> list[tuple[str, int, str]]:
    findings = []
    for relative in sorted(paths):
        path = root / relative
        if path.is_symlink():
            findings.append((relative, 0, "symlink"))
            continue
        parts = Path(relative).parts
        if any(part in RUNTIME_DIRS or part.startswith(".venv") for part in parts) or path.suffix in RUNTIME_SUFFIXES:
            findings.append((relative, 0, "runtime file"))
            continue
        if path.name == ".env" or path.name.startswith(".env.") or path.suffix in (".pem", ".key"):
            findings.append((relative, 0, "credential file"))
            continue
        # Publication source is UTF-8 text. Binary artifacts are outside the release.
        raw = path.read_bytes()
        if b"\x00" in raw:
            findings.append((relative, 0, "binary artifact"))
            continue
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            for category, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match is None:
                    continue
                if category == "literal API key" and match.group(1).startswith(("mock", "fixture", "test", "$", "<")):
                    continue
                findings.append((relative, number, category))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, help="NUL-separated Git paths; omit for a clean source export.")
    args = parser.parse_args()
    paths = ([path for path in args.file_list.read_text().split("\0") if path] if args.file_list else
             [path.relative_to(args.root).as_posix() for path in args.root.rglob("*") if path.is_file() or path.is_symlink()])
    findings = check_files(args.root, paths)
    for path, number, category in findings:
        print(f"{path}:{number}: {category}")
    print(f"Checked {len(paths)} publication files; {len(findings)} findings.")
    return int(bool(findings))


if __name__ == "__main__":
    sys.exit(main())
