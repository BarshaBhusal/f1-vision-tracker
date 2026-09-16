"""
src/identity_seed.py

Purpose
-------
Manually seed real driver identity onto track_ids using the known,
official starting grid order (Section 14 of the spec explicitly
allows "manual mapping for controlled footage" as a legitimate
identity method - this isn't a workaround, it's a sanctioned
approach for exactly this situation).

Why manual, not automatic: the grid is two staggered columns viewed
from an angled broadcast camera. There's no safe, general rule like
"leftmost box = P1" - it needs one human glance to match boxes to
grid slots correctly. This script makes that one-time glance fast:
it shows every track_id's actual cropped image from the grid-start
frame, side by side, so you can quickly note "track_id 3 = the
Ferrari" etc. It does NOT guess the mapping for you.

Two steps:
    1. `--generate` : produces an HTML sheet with each track_id's
       crop image, and the official grid list for reference, so you
       can write down the mapping.
    2. `--apply`    : once you've filled in a mapping CSV (track_id,
       driver), joins it onto a tracks CSV, adding driver/team
       columns for every row of that track from then on.

IMPORTANT LIMITATION (inherited from Phase 4): this mapping is only
as good as tracking continuity. If a track_id later dies and
respawns as a new ID (dropout), or swaps with a neighboring car
(occlusion), the seeded identity does NOT automatically follow -
it stays attached to the original track_id only. This seeds identity
correctly at the start; it doesn't fix Phase 4's remaining tracking
limitations.

Usage:
    # Step 1: generate the visual matching sheet from the grid-start frame
    python src/identity_seed.py --generate --video data/videos/F1_FullVideo.mp4 \\
        --tracks-csv outputs/tracking/tracks_F1_FullVideo.csv

    # (open outputs/analytics/grid_matching_sheet.html, look at each crop,
    #  fill in outputs/analytics/grid_mapping.csv by hand: track_id,driver)

    # Step 2: apply your filled-in mapping to the tracks CSV
    python src/identity_seed.py --apply --tracks-csv outputs/tracking/tracks_F1_FullVideo.csv \\
        --mapping outputs/analytics/grid_mapping.csv
"""

import argparse
import base64
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_OUT_DIR = PROJECT_ROOT / "outputs" / "analytics"
GRID_REFERENCE_PATH = PROJECT_ROOT / "configs" / "canadian_gp_2026_grid.json"


def load_tracks_by_frame(csv_path: Path) -> dict:
    by_frame = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(row)
    return by_frame


def get_first_appearance_crops(video_path: Path, tracks_by_frame: dict) -> dict:
    """
    For each track_id, grab its crop from the EARLIEST frame it
    appears in (closest to the true grid-start moment), not just any
    frame - we want the cleanest, most representative view for
    matching against the grid list.
    """
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
            crops[tid] = (frame_idx, crop)

    cap.release()
    return crops


def crop_to_base64(crop) -> str:
    ok, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf).decode("utf-8") if ok else ""


def build_full_frame_overview(video_path: Path, tracks_by_frame: dict, frame_idx: int = 0) -> str:
    """
    Draw the FULL frame (not individual crops) with every track_id's
    box and label visible together. Isolated crops lose row/position
    context that's often the actual disambiguating signal for a grid
    start (e.g. "front row, left side" narrows it to 1-2 candidates
    even when the crop itself is too small/blurry to read livery
    detail). This gives that context back.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return ""

    for row in tracks_by_frame.get(frame_idx, []):
        x1, y1, x2, y2 = (int(float(row["x1"])), int(float(row["y1"])),
                           int(float(row["x2"])), int(float(row["y2"])))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, row["track_id"], (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf).decode("utf-8")


def generate_matching_sheet(video_path: Path, tracks_csv: Path):
    tracks_by_frame = load_tracks_by_frame(tracks_csv)
    crops = get_first_appearance_crops(video_path, tracks_by_frame)
    earliest_frame = min(tracks_by_frame.keys()) if tracks_by_frame else 0
    full_frame_b64 = build_full_frame_overview(video_path, tracks_by_frame, earliest_frame)

    grid_data = {}
    if GRID_REFERENCE_PATH.exists():
        with open(GRID_REFERENCE_PATH, "r") as f:
            grid_data = json.load(f)

    grid_rows = "".join(
        f"<tr><td>{g['position']}</td><td>{g['driver']}</td><td>{g['team']}</td></tr>"
        for g in grid_data.get("grid", [])
    )

    track_tiles = "".join(
        f"""<div class="tile">
                <img src="data:image/jpeg;base64,{crop_to_base64(crop)}">
                <div class="tid">track_id: {tid}</div>
                <div class="frame">first seen: frame {frame_idx}</div>
            </div>"""
        for tid, (frame_idx, crop) in sorted(crops.items(), key=lambda x: int(x[0]))
    )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Grid Matching Sheet</title>
    <style>
        body {{ font-family: sans-serif; background: #12141a; color: #eee; padding: 24px; }}
        h2 {{ color: #9ab; }}
        .tiles {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 32px; }}
        .tile {{ background: #1c1f26; padding: 8px; border-radius: 8px; text-align: center; }}
        .tile img {{ max-width: 160px; display: block; border-radius: 4px; }}
        .tid {{ font-weight: bold; margin-top: 6px; font-size: 13px; }}
        .frame {{ font-size: 11px; color: #888; }}
        table {{ border-collapse: collapse; }}
        td {{ border: 1px solid #333; padding: 4px 10px; font-size: 13px; }}
        .note {{ color: #e6a700; font-size: 13px; margin-bottom: 16px; }}
        .full-frame {{ max-width: 100%; border-radius: 8px; border: 1px solid #333; margin-bottom: 24px; }}
    </style></head><body>
    <h2>Full frame with all track_id boxes (use position/row context to help identify)</h2>
    <img class="full-frame" src="data:image/jpeg;base64,{full_frame_b64}">
    <h2>Step 1: Look at each track_id's crop below</h2>
    <p class="note">Match each crop to a driver in the grid reference table, then fill in
    outputs/analytics/grid_mapping.csv by hand: one row per track_id, e.g. "3,Lando Norris"</p>
    <div class="tiles">{track_tiles}</div>
    <h2>Official Grid Reference ({grid_data.get('race', 'unknown race')})</h2>
    <table><tr><th>Pos</th><th>Driver</th><th>Team</th></tr>{grid_rows}</table>
    </body></html>"""

    ANALYTICS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_path = ANALYTICS_OUT_DIR / "grid_matching_sheet.html"
    with open(sheet_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Also write a blank template CSV, ready to be filled in
    template_path = ANALYTICS_OUT_DIR / "grid_mapping.csv"
    if not template_path.exists():
        with open(template_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "driver"])
            for tid in sorted(crops.keys(), key=int):
                writer.writerow([tid, ""])

    print(f"Matching sheet: {sheet_path}")
    print(f"Fill in this template by hand: {template_path}")
    print("(open the HTML sheet in a browser, then edit the CSV with your best match per track_id)")


def apply_mapping(tracks_csv: Path, mapping_csv: Path):
    with open(mapping_csv, "r", newline="") as f:
        mapping = {row["track_id"]: row["driver"] for row in csv.DictReader(f) if row["driver"].strip()}

    if not mapping:
        print("WARNING: mapping CSV has no filled-in driver names yet - nothing to apply.")
        return

    grid_data = {}
    if GRID_REFERENCE_PATH.exists():
        with open(GRID_REFERENCE_PATH, "r") as f:
            grid_data = json.load(f)
    team_by_driver = {g["driver"]: g["team"] for g in grid_data.get("grid", [])}

    with open(tracks_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if "driver" not in fieldnames:
        fieldnames += ["driver", "team"]

    matched = 0
    for row in rows:
        driver = mapping.get(row["track_id"])
        row["driver"] = driver or ""
        row["team"] = team_by_driver.get(driver, "") if driver else ""
        if driver:
            matched += 1

    out_path = tracks_csv.with_name(tracks_csv.stem + "_with_identity.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Applied identity mapping to {matched}/{len(rows)} rows.")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Seed driver identity onto track_ids using the known starting grid.")
    parser.add_argument("--generate", action="store_true", help="Generate the visual matching sheet + blank mapping template.")
    parser.add_argument("--apply", action="store_true", help="Apply a filled-in mapping CSV to a tracks CSV.")
    parser.add_argument("--video", type=str, help="Video path (required with --generate).")
    parser.add_argument("--tracks-csv", type=str, required=True, help="Path to tracker.py's tracks CSV.")
    parser.add_argument("--mapping", type=str, help="Path to filled-in mapping CSV (required with --apply).")
    args = parser.parse_args()

    tracks_csv = Path(args.tracks_csv)
    if not tracks_csv.exists():
        print(f"ERROR: {tracks_csv} does not exist.")
        sys.exit(1)

    if args.generate:
        if not args.video:
            print("ERROR: --video required with --generate.")
            sys.exit(1)
        generate_matching_sheet(Path(args.video), tracks_csv)
    elif args.apply:
        if not args.mapping:
            print("ERROR: --mapping required with --apply.")
            sys.exit(1)
        apply_mapping(tracks_csv, Path(args.mapping))
    else:
        print("ERROR: specify --generate or --apply.")
        sys.exit(1)


if __name__ == "__main__":
    main()