"""
src/leaderboard_ocr_test.py

Purpose
-------
car-number OCR (ocr_test.py) failed completely: 0/13 crops readable.
That target was a small, motion-blurred, photographed object.

The broadcast leaderboard graphic is a different kind of target
entirely: high-contrast, computer-rendered text, not a photograph of
a physical object. We confirmed one appears around frame 3000 in
F1_FullVideo.mp4 (seen during Phase 3 testing - driver abbreviations
and gaps listed down the left side of the frame). This script tests
OCR specifically on that region, on that known frame, rather than
guessing across the whole video blind.

If this works: repeated, independent driver-position checkpoints
throughout the race - a much stronger identity signal than the
one-time grid-matching attempt, since it doesn't depend on tracking
continuity at all (reads position directly from the broadcast graphic
every time it's visible).

Usage:
    python src/leaderboard_ocr_test.py --video data/videos/F1_FullVideo.mp4 --frame 3000

    # If the crop region needs adjusting (leaderboard isn't fully captured):
    python src/leaderboard_ocr_test.py --video data/videos/F1_FullVideo.mp4 --frame 3000 \\
        --x-frac 0.30 --y-frac 1.0
"""

import argparse
import sys
from pathlib import Path

import cv2

try:
    import easyocr
except ImportError:
    print("ERROR: easyocr not installed. Run: pip install easyocr")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OCR_TEST_OUT_DIR = PROJECT_ROOT / "outputs" / "ocr_test"


def get_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read frame {frame_idx}")
    return frame


def crop_leaderboard_region(frame, x_frac: float, y_frac: float):
    """
    Crop the top-left portion of the frame, where the leaderboard
    graphic was visible in our earlier testing. x_frac/y_frac control
    how much of the width/height to include - defaults are a
    starting guess, adjust via CLI if the graphic isn't fully
    captured (or too much surrounding footage is included).
    """
    h, w = frame.shape[:2]
    crop_w = int(w * x_frac)
    crop_h = int(h * y_frac)
    return frame[0:crop_h, 0:crop_w]


def main():
    parser = argparse.ArgumentParser(description="Test OCR on the broadcast leaderboard graphic.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--frame", type=int, default=3000,
                         help="Frame index known to show the leaderboard (default: 3000).")
    parser.add_argument("--x-frac", type=float, default=0.25,
                         help="Fraction of frame width to crop from the left (default 0.25).")
    parser.add_argument("--y-frac", type=float, default=1.0,
                         help="Fraction of frame height to crop from the top (default 1.0 = full height).")
    parser.add_argument("--upscale", type=float, default=3.0,
                         help="Upscale factor before OCR (default 3.0).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: {video_path} does not exist.")
        sys.exit(1)

    print(f"Reading frame {args.frame} from {video_path.name} ...")
    frame = get_frame(video_path, args.frame)

    crop = crop_leaderboard_region(frame, args.x_frac, args.y_frac)
    OCR_TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_crop_path = OCR_TEST_OUT_DIR / f"leaderboard_crop_frame{args.frame}.jpg"
    cv2.imwrite(str(raw_crop_path), crop)
    print(f"Saved raw crop (before OCR) to: {raw_crop_path}")
    print("  -> Check this image first: does it actually contain the leaderboard?")
    print("     If not, adjust --x-frac / --y-frac and rerun.")

    upscaled = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)

    print("\nLoading EasyOCR reader (first run downloads a model)...")
    reader = easyocr.Reader(['en'], gpu=False)

    print("Running OCR on the leaderboard crop...")
    detections = reader.readtext(upscaled)

    print(f"\n{'=' * 60}")
    print("LEADERBOARD OCR RESULTS")
    print(f"{'=' * 60}")
    if not detections:
        print("  Nothing detected. Either the crop region is wrong (check the")
        print("  saved crop image) or the graphic still isn't OCR-readable here.")
    else:
        for (bbox, text, conf) in detections:
            print(f"  '{text}'  (confidence: {conf:.2f})")

        annotated = upscaled.copy()
        for (bbox, text, conf) in detections:
            top_left = (int(bbox[0][0]), int(bbox[0][1]))
            cv2.putText(annotated, text, top_left, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        annotated_path = OCR_TEST_OUT_DIR / f"leaderboard_ocr_frame{args.frame}.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        print(f"\nAnnotated result saved to: {annotated_path}")


if __name__ == "__main__":
    main()