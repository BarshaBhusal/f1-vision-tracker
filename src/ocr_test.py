"""
src/ocr_test.py

Purpose
-------
One quick, honest test: can we read car numbers off the cars in our
footage? This is a genuinely different identity signal than color
(Phase 5's identity.py) - numbers are high-contrast and distinctive
even between teammates, where color fails completely.

This is NOT meant to become a full pipeline yet. It's a fast check:
run OCR on every tracked car crop from the grid-start frame, print
what it finds (if anything), and save annotated crops so we can
SEE the result rather than just trust a confidence number.

If this works: real, reliable identity - worth building out properly.
If this doesn't work: honest evidence that OCR isn't viable at this
resolution, and identity mapping should be documented as a known
limitation rather than pursued further.

Setup:
    pip install easyocr
    (first run downloads a small recognition model, needs internet once)

Usage:
    python src/ocr_test.py --video data/videos/F1_FullVideo.mp4 \\
        --tracks-csv outputs/tracking/tracks_F1_FullVideo.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

try:
    import easyocr
except ImportError:
    print("ERROR: easyocr not installed. Run: pip install easyocr")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OCR_TEST_OUT_DIR = PROJECT_ROOT / "outputs" / "ocr_test"


def load_tracks_by_frame(csv_path: Path) -> dict:
    by_frame = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(row)
    return by_frame


def get_first_appearance_crops(video_path: Path, tracks_by_frame: dict) -> dict:
    """Same approach as identity_seed.py - earliest, cleanest crop per track."""
    first_frame_for_track = {}
    for frame_idx in sorted(tracks_by_frame.keys()):
        for row in tracks_by_frame[frame_idx]:
            tid = row["track_id"]
            if tid not in first_frame_for_track:
                first_frame_for_track[tid] = (frame_idx, row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    crops = {}
    for tid, (frame_idx, row) in first_frame_for_track.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = (int(float(row["x1"])), int(float(row["y1"])),
                           int(float(row["x2"])), int(float(row["y2"])))
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops[tid] = crop

    cap.release()
    return crops


def run_ocr_test(crops: dict, reader) -> dict:
    """
    Run OCR on each crop, upscaled 4x (small source crops need this -
    same principle as the --upscale flag we added to detector.py).
    Returns {track_id: [(text, confidence), ...]}.
    """
    results = {}
    OCR_TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tid, crop in crops.items():
        upscaled = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        detections = reader.readtext(upscaled)

        results[tid] = [(text, round(float(conf), 3)) for (_, text, conf) in detections]

        # Save annotated crop for visual inspection - just overlay the
        # detected text as a label rather than drawing exact polygons,
        # keeps this simple since it's a quick test, not production code
        annotated = upscaled.copy()
        label = "; ".join(f"{text}({conf:.2f})" for text, conf in results[tid]) or "(none)"
        cv2.putText(annotated, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out_path = OCR_TEST_OUT_DIR / f"track_{tid}.jpg"
        cv2.imwrite(str(out_path), annotated)

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick test: can we OCR car numbers?")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--tracks-csv", type=str, required=True)
    args = parser.parse_args()

    video_path = Path(args.video)
    tracks_csv = Path(args.tracks_csv)
    if not video_path.exists() or not tracks_csv.exists():
        print("ERROR: video or tracks CSV not found.")
        sys.exit(1)

    tracks_by_frame = load_tracks_by_frame(tracks_csv)
    crops = get_first_appearance_crops(video_path, tracks_by_frame)
    print(f"Testing OCR on {len(crops)} track crop(s)...")

    print("Loading EasyOCR reader (first run downloads a model, ~1-2 min)...")
    reader = easyocr.Reader(['en'], gpu=False)  # CPU-only, per our hardware constraints

    results = run_ocr_test(crops, reader)

    print(f"\n{'=' * 60}")
    print("OCR RESULTS")
    print(f"{'=' * 60}")
    any_hits = False
    for tid, detections in results.items():
        if detections:
            any_hits = True
            print(f"  track_id {tid}: {detections}")
        else:
            print(f"  track_id {tid}: (nothing detected)")

    print(f"\nAnnotated crops saved to: {OCR_TEST_OUT_DIR}")
    if not any_hits:
        print("\nNo text detected on ANY crop - likely means car numbers are not")
        print("readable at this resolution. This is a real, useful negative result.")
    else:
        print("\nSome text detected - check the annotated crops visually to see")
        print("if it's actually a car number, or a false positive (sponsor logo, etc).")


if __name__ == "__main__":
    main()