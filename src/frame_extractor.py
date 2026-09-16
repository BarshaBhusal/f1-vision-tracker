"""
src/frame_extractor.py

Phase 2 - Frame Extraction
---------------------------
Purpose:
    A video is fundamentally a sequence of frames. This module provides
    the reusable building block every later phase depends on: a way to
    access individual frames from a video, either as a stream (for
    detection/tracking, which should NOT require pre-saving every
    frame to disk) or saved to disk (for visual inspection / dataset
    building).

Why frame-index-based, not time-based:
    video_info.py showed our clips have inconsistent FPS (49.6-56.19
    across clips, non-standard values). This is consistent with
    variable-frame-rate (VFR) source files. Timestamp-based sampling
    (e.g. "grab a frame every 0.5 seconds") would drift or behave
    unpredictably on VFR video. Frame-index-based sampling ("grab
    every Nth frame") is exact regardless of FPS irregularities, so
    that's what this module uses. Timestamps are still recorded
    per-frame (CAP_PROP_POS_MSEC) for later reference, but treated as
    approximate, not authoritative.

Two entry points:
    1. iter_frames()          -> generator, used by Phase 3/4 pipelines
                                  to stream frames without disk I/O.
    2. extract_frames_to_disk() -> saves a sampled set of frames as
                                  JPGs, for visual inspection or
                                  building a small labeled dataset.

Usage (standalone / CLI):
    # Stream through a video and just report frame count / basic stats
    python src/frame_extractor.py --video data/videos/F1_FullVideo.mp4 --report-only

    # Save every 25th frame to disk
    python src/frame_extractor.py --video data/videos/F1_FullVideo.mp4 --interval 25

    # Save exactly 20 evenly-spaced frames
    python src/frame_extractor.py --video data/videos/F1_FullVideo.mp4 --count 20
"""

import argparse
import sys
from pathlib import Path
from typing import Generator, NamedTuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_OUT_DIR = PROJECT_ROOT / "outputs" / "frames"


class FrameData(NamedTuple):
    """One frame plus the metadata every downstream phase will need."""
    index: int          # frame index in the video (0-based)
    timestamp_ms: float  # approximate, per VFR caveat above
    frame: np.ndarray   # BGR image array (OpenCV default)


def iter_frames(video_path: Path, frame_skip: int = 1) -> Generator[FrameData, None, None]:
    """
    Stream frames directly from the video, no disk writes.

    This is the function Phase 3 (detection) and Phase 4 (tracking)
    should import and loop over - it's the actual production path.
    extract_frames_to_disk() below is for inspection only.

    Args:
        video_path: path to the video file.
        frame_skip: yield every Nth frame (1 = every frame).
                    Useful for faster iteration during development,
                    or on CPU-only hardware per Section 22.

    Yields:
        FrameData(index, timestamp_ms, frame)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video, or an unreadable frame
            if idx % frame_skip == 0:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                yield FrameData(index=idx, timestamp_ms=timestamp_ms, frame=frame)
            idx += 1
    finally:
        cap.release()


def extract_frames_to_disk(
    video_path: Path,
    output_dir: Path = None,
    interval: int = None,
    count: int = None,
) -> list:
    """
    Save a sampled set of frames as JPGs for visual inspection.

    Exactly one of `interval` or `count` should be given:
        interval=N  -> save every Nth frame (variable total count)
        count=N     -> save N evenly-spaced frames across the video

    Returns list of saved file paths.
    """
    if interval is None and count is None:
        raise ValueError("Provide either interval or count.")
    if interval is not None and count is not None:
        raise ValueError("Provide only one of interval or count, not both.")

    if output_dir is None:
        output_dir = FRAMES_OUT_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        raise IOError(f"Video reports zero frames: {video_path}")

    if count is not None:
        n = max(1, min(count, total_frames))
        target_indices = set(int(total_frames * (i + 1) / (n + 1)) for i in range(n))
    else:
        target_indices = set(range(0, total_frames, interval))

    saved_paths = []
    for fd in iter_frames(video_path, frame_skip=1):
        if fd.index in target_indices:
            out_path = output_dir / f"frame_{fd.index:06d}.jpg"
            cv2.imwrite(str(out_path), fd.frame)
            saved_paths.append(out_path)
        if fd.index >= max(target_indices):
            break

    return saved_paths


def report_only(video_path: Path) -> None:
    """
    Stream through the whole video without saving anything, just to
    confirm every frame is readable end-to-end and report basic
    stats. Useful sanity check before building the detection stage -
    catches partially-corrupted files that opened fine but fail
    mid-stream.
    """
    total = 0
    failed_reads = 0
    widths, heights = set(), set()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return
    expected_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for fd in iter_frames(video_path, frame_skip=1):
        total += 1
        h, w = fd.frame.shape[:2]
        widths.add(w)
        heights.add(h)

    failed_reads = expected_total - total

    print(f"\n{video_path.name}")
    print(f"  Frames expected (metadata): {expected_total}")
    print(f"  Frames actually readable:   {total}")
    if failed_reads > 0:
        print(f"  WARNING: {failed_reads} frame(s) reported by metadata but not readable.")
    print(f"  Frame width(s) seen:  {sorted(widths)}")
    print(f"  Frame height(s) seen: {sorted(heights)}")
    if len(widths) > 1 or len(heights) > 1:
        print("  WARNING: frame dimensions are not consistent within this video.")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Frame extraction / inspection.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    parser.add_argument("--interval", type=int, default=None, help="Save every Nth frame.")
    parser.add_argument("--count", type=int, default=None, help="Save N evenly-spaced frames.")
    parser.add_argument("--report-only", action="store_true", help="Stream + validate readability, save nothing.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: file not found: {video_path}")
        sys.exit(1)

    if args.report_only:
        report_only(video_path)
        return

    if args.interval is None and args.count is None:
        print("ERROR: provide --interval N, --count N, or --report-only.")
        sys.exit(1)

    saved = extract_frames_to_disk(video_path, interval=args.interval, count=args.count)
    print(f"Saved {len(saved)} frame(s) to {FRAMES_OUT_DIR / video_path.stem}/")


if __name__ == "__main__":
    main()