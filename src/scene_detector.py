"""
src/scene_detector.py

Purpose
-------
F1_FullVideo.mp4 is a concatenation of 10 separately-filmed clips.
That means it likely contains hard camera cuts - moments where the
video jumps from one shot/camera/moment to a completely different
one. No tracker (ByteTrack, BoT-SORT, or otherwise) can maintain a
car's identity across a hard cut - that's not a tracking failure,
it's an unavoidable consequence of the footage itself.

This script flags likely cut locations using frame-to-frame color
histogram difference: consecutive frames from continuous footage look
similar (small diff), while a cut produces a large, sudden jump.

This isn't just a one-off diagnostic - Phase 6/7 (track/lap
understanding) needs to know where camera cuts are anyway, since a
car's position/trajectory logic shouldn't be treated as continuous
across a cut. Worth having as reusable infrastructure now.

Usage:
    python src/scene_detector.py --video data/videos/F1_FullVideo.mp4

    # Check whether a specific frame range contains a cut
    python src/scene_detector.py --video data/videos/F1_FullVideo.mp4 --check-range 2900 3200
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_OUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def compute_histogram(frame) -> np.ndarray:
    """3D color histogram, normalized, for comparing frame similarity."""
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def detect_cuts(video_path: Path, threshold: float = 0.5) -> list:
    """
    Walk through the video comparing each frame's histogram to the
    previous one. A correlation below `threshold` (1.0 = identical,
    0.0 = totally different) flags a likely cut.

    Returns list of dicts: {frame_index, timestamp_ms, similarity}
    for every flagged cut.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    cuts = []
    prev_hist = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hist = compute_histogram(frame)
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < threshold:
                cuts.append({
                    "frame_index": idx,
                    "timestamp_ms": round(cap.get(cv2.CAP_PROP_POS_MSEC), 1),
                    "similarity": round(float(similarity), 4),
                })
        prev_hist = hist
        idx += 1

    cap.release()
    return cuts


def main():
    parser = argparse.ArgumentParser(description="Detect likely camera cuts via histogram differencing.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Similarity threshold below which a frame transition is flagged as a cut (0-1, lower = stricter).")
    parser.add_argument("--check-range", type=int, nargs=2, metavar=("START", "END"), default=None,
                         help="After detection, report whether any cut falls within this frame range.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: {video_path} does not exist.")
        sys.exit(1)

    print(f"Scanning {video_path.name} for likely camera cuts (threshold={args.threshold}) ...")
    cuts = detect_cuts(video_path, args.threshold)

    print(f"\nFound {len(cuts)} likely cut(s):")
    for cut in cuts:
        print(f"  frame {cut['frame_index']:6d}  (t={cut['timestamp_ms']/1000:.2f}s)  "
              f"similarity={cut['similarity']}")

    REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_OUT_DIR / f"scene_cuts_{video_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump({"video": str(video_path), "threshold": args.threshold, "cuts": cuts}, f, indent=2)
    print(f"\nSaved to: {out_path}")

    if args.check_range:
        start, end = args.check_range
        in_range = [c for c in cuts if start <= c["frame_index"] <= end]
        print(f"\nCuts within frame range [{start}, {end}]:")
        if in_range:
            for c in in_range:
                print(f"  frame {c['frame_index']} - likely a hard cut in this range.")
        else:
            print("  None found - any tracking instability here is NOT explained by a camera cut.")


if __name__ == "__main__":
    main()