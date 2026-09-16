"""
src/video_info.py

Phase 1 — Video Understanding
------------------------------
Purpose:
    Before touching YOLO, tracking, or any deep learning, this script
    establishes the fundamental relationship:

        VIDEO -> FRAMES -> COMPUTER VISION

    It answers the basic questions every downstream phase depends on:
        - How long is the footage?
        - What resolution / frame rate are we working with?
        - Can OpenCV even open the file cleanly?
        - What does a representative frame actually look like?

Usage:
    # Inspect the primary development video (default)
    python src/video_info.py

    # Inspect a specific file
    python src/video_info.py --video data/videos/clip_03.mp4

    # Inspect every video in data/videos/ (full video + all 10 clips)
    python src/video_info.py --all

    # Also save N representative frames to outputs/frames/<video_name>/
    python src/video_info.py --save-frames 5

Output:
    - Printed metadata report per video
    - JSON report written to outputs/reports/video_info.json
    - (optional) sampled frame images written to outputs/frames/<video_name>/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Project paths (relative to project root, not machine-specific)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "videos" / "F1_FullVideo.mp4"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
FRAMES_OUT_DIR = PROJECT_ROOT / "outputs" / "frames"
REPORTS_OUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def inspect_video(video_path: Path) -> dict:
    """
    Open a video file with OpenCV and extract its core metadata.

    Returns a dict so the result is easy to print, log, or serialize to
    JSON/CSV for later phases. Raises no exceptions on a bad file -
    instead returns a dict with an 'error' key, so batch runs (--all)
    don't crash on one corrupted clip.
    """
    result = {
        "file": str(video_path),
        "exists": video_path.exists(),
        "opened_successfully": False,
        "fps": None,
        "total_frames": None,
        "width": None,
        "height": None,
        "duration_seconds": None,
        "duration_hms": None,
        "error": None,
    }

    if not video_path.exists():
        result["error"] = "File not found."
        return result

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        result["error"] = "OpenCV could not open this file (corrupted or unsupported codec)."
        cap.release()
        return result

    result["opened_successfully"] = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = (total_frames / fps) if fps and fps > 0 else None

    result["fps"] = round(fps, 3) if fps else None
    result["total_frames"] = total_frames
    result["width"] = width
    result["height"] = height
    result["duration_seconds"] = round(duration, 3) if duration else None
    result["duration_hms"] = _seconds_to_hms(duration) if duration else None

    cap.release()
    return result


def _seconds_to_hms(seconds: float) -> str:
    """Human-readable duration, e.g. 623.4 -> '0:10:23.4'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def save_representative_frames(video_path: Path, n_frames: int) -> list:
    """
    Sample n_frames evenly across the video and save them as JPGs so we
    can visually inspect car size, camera angle, lighting, occlusion,
    motion blur, etc. before designing the detection stage.

    Frames are NOT extracted exhaustively - this is inspection, not
    training-data generation (that comes later, if needed, in Phase 2/3).
    """
    saved_paths = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return saved_paths

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return saved_paths

    n_frames = max(1, min(n_frames, total_frames))
    # Evenly spaced frame indices, avoiding the very first/last frame
    indices = [int(total_frames * (i + 1) / (n_frames + 1)) for i in range(n_frames)]

    out_dir = FRAMES_OUT_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = out_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_paths.append(str(out_path))

    cap.release()
    return saved_paths


def print_report(result: dict) -> None:
    """Print a clean, readable metadata report for one video."""
    print(f"\n{'=' * 60}")
    print(f"VIDEO: {result['file']}")
    print(f"{'=' * 60}")

    if not result["exists"]:
        print("  STATUS: NOT FOUND")
        return

    if not result["opened_successfully"]:
        print(f"  STATUS: FAILED TO OPEN ({result['error']})")
        return

    print(f"  STATUS:          OK")
    print(f"  FPS:              {result['fps']}")
    print(f"  Total Frames:     {result['total_frames']}")
    print(f"  Resolution:       {result['width']} x {result['height']}")
    print(f"  Duration:         {result['duration_seconds']} sec ({result['duration_hms']})")


def resolve_targets(args) -> list:
    """Determine which video file(s) to process based on CLI args."""
    if args.all:
        if not VIDEOS_DIR.exists():
            print(f"ERROR: {VIDEOS_DIR} does not exist.")
            sys.exit(1)
        videos = sorted(VIDEOS_DIR.glob("*.mp4"))
        if not videos:
            print(f"ERROR: No .mp4 files found in {VIDEOS_DIR}.")
            sys.exit(1)
        return videos

    video_path = Path(args.video) if args.video else DEFAULT_VIDEO
    return [video_path]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Inspect FPS, frame count, resolution, and duration of F1 race footage."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=f"Path to a single video file (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect every .mp4 in data/videos/ (F1_FullVideo.mp4 + all 10 clips).",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=0,
        metavar="N",
        help="Save N evenly-spaced representative frames per video to outputs/frames/<video_name>/",
    )
    args = parser.parse_args()

    targets = resolve_targets(args)
    all_results = []

    for video_path in targets:
        result = inspect_video(video_path)
        print_report(result)
        all_results.append(result)

        if result["opened_successfully"] and args.save_frames > 0:
            saved = save_representative_frames(video_path, args.save_frames)
            print(f"  Saved {len(saved)} representative frame(s) to {FRAMES_OUT_DIR / video_path.stem}/")

    # Persist a structured JSON report for later phases to reference
    REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_OUT_DIR / "video_info.json"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "videos": all_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Report written to: {report_path}")
    print(f"{'=' * 60}\n")

    # Non-zero exit if anything failed to open, useful for CI / scripting
    if any(not r["opened_successfully"] for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()