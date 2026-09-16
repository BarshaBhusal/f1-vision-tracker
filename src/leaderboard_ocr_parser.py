"""
src/leaderboard_ocr_parser.py

Purpose
-------
leaderboard_ocr_test.py proved the concept: raw OCR on the broadcast
leaderboard produces recognizable (if garbled) driver code fragments
- 'VcR' for VER, 'RUS Ireryai' containing 'RUS', etc. Zero hits on
car-number OCR, but real signal here.

This script cleans that up. We know the EXACT, finite set of valid
3-letter FIA driver codes for this race (from the grid reference
JSON) - so instead of trusting raw OCR text, we fuzzy-match each
detected fragment against that known list. 'VcR' is 1 letter off
from 'VER' - close enough to confidently accept. Random noise like
'Wut' or '{iut' won't be close to any real code and gets discarded.

This is a much stronger approach than trusting OCR text directly:
we're not asking "what does this say" (hard, error-prone), we're
asking "which of these ~22 known possibilities does this most
resemble" (much easier, bounded problem).

Usage:
    python src/leaderboard_ocr_parser.py --video data/videos/F1_FullVideo.mp4 --frame 3000
"""

import argparse
import json
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
GRID_REFERENCE_PATH = PROJECT_ROOT / "configs" / "canadian_gp_2026_grid.json"

# Standard FIA 3-letter codes for the drivers in our grid reference.
# (The grid JSON has full names; codes are the broadcast convention.)
DRIVER_CODES = {
    "RUS": "George Russell", "ANT": "Kimi Antonelli", "NOR": "Lando Norris",
    "PIA": "Oscar Piastri", "HAM": "Lewis Hamilton", "VER": "Max Verstappen",
    "HAD": "Isack Hadjar", "LEC": "Charles Leclerc", "LIN": "Arvid Lindblad",
    "COL": "Franco Colapinto", "HUL": "Nico Hulkenberg", "LAW": "Liam Lawson",
    "BOR": "Gabriel Bortoleto", "GAS": "Pierre Gasly", "SAI": "Carlos Sainz",
    "BEA": "Ollie Bearman", "OCO": "Esteban Ocon", "ALB": "Alex Albon",
    "ALO": "Fernando Alonso", "PER": "Sergio Perez", "BOT": "Valtteri Bottas",
    "STR": "Lance Stroll",
}


def edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance, no external dependency needed."""
    if len(a) < len(b):
        return edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = range(len(b) + 1)
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def best_code_match(raw_text: str, max_distance: int = 1):
    """
    Try every 3-letter (and slightly longer, for OCR noise) window of
    the raw text against every known code, return the closest match
    within max_distance, or None if nothing is close enough.
    """
    cleaned = "".join(c for c in raw_text.upper() if c.isalpha())
    if len(cleaned) < 2:
        return None

    best = None
    best_dist = max_distance + 1

    # Try the whole cleaned string, and also 3-letter windows within it
    # (raw text sometimes has extra garbage attached, e.g. "RUSIreryai")
    candidates = [cleaned]
    for i in range(len(cleaned) - 2):
        candidates.append(cleaned[i:i + 3])

    for candidate in candidates:
        for code in DRIVER_CODES:
            dist = edit_distance(candidate, code)
            if dist < best_dist:
                best_dist = dist
                best = code

    return best if best_dist <= max_distance else None


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


def main():
    parser = argparse.ArgumentParser(description="Fuzzy-match leaderboard OCR output against known driver codes.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--frame", type=int, default=3000)
    parser.add_argument("--x-frac", type=float, default=0.25)
    parser.add_argument("--y-frac", type=float, default=1.0)
    parser.add_argument("--upscale", type=float, default=3.0)
    parser.add_argument("--max-distance", type=int, default=1,
                         help="Max edit distance to accept a fuzzy match (default 1).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: {video_path} does not exist.")
        sys.exit(1)

    frame = get_frame(video_path, args.frame)
    h, w = frame.shape[:2]
    crop = frame[0:int(h * args.y_frac), 0:int(w * args.x_frac)]
    upscaled = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)

    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)

    print("Running OCR...")
    detections = reader.readtext(upscaled)

    # Sort top-to-bottom by vertical position - this preserves the
    # leaderboard's displayed order (1st, 2nd, 3rd...) as a proxy for
    # running position at this moment in the race.
    detections.sort(key=lambda d: d[0][0][1])

    matches = []
    for (bbox, text, conf) in detections:
        matched_code = best_code_match(text, args.max_distance)
        if matched_code:
            matches.append({
                "raw_text": text,
                "matched_code": matched_code,
                "driver": DRIVER_CODES[matched_code],
                "ocr_confidence": round(float(conf), 3),
            })

    print(f"\n{'=' * 60}")
    print(f"MATCHED DRIVER CODES (top-to-bottom order, frame {args.frame})")
    print(f"{'=' * 60}")
    if not matches:
        print("  No confident matches found.")
    else:
        for i, m in enumerate(matches, 1):
            print(f"  {i}. '{m['raw_text']}' -> {m['matched_code']} ({m['driver']}) "
                  f"[OCR confidence: {m['ocr_confidence']}]")

    print(f"\n{len(matches)} of {len(detections)} raw OCR detections matched a known driver code.")
    print("\nNOTE: order reflects the on-screen leaderboard at this moment, which")
    print("may be RACE POSITION, not necessarily grid position - verify against")
    print("context (e.g. this frame showed a mid-race incident, not the start).")


if __name__ == "__main__":
    main()