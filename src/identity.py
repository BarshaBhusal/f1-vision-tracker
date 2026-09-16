"""
src/identity.py

Phase 5 - Car Identity (starting point: livery/color signature)
------------------------------------------------------------------
Purpose:
    track_id from Phase 4 is NOT the same thing as knowing which car
    this actually is (Section 14 of the spec). track_id can also
    reset when a track is lost and re-detected (Phase 4's known
    dropout/occlusion issues), so relying on it alone under-counts
    unique cars and over-counts "new" ones.

    This script adds a car-level signal that's independent of
    track_id: each car's average livery color, sampled from its
    bounding box crops across the frames it appears in. This is
    the most resolution-tolerant identity signal available given our
    848x384 footage (OCR on car numbers is a separate, harder
    experiment worth trying next, not first, given resolution).

Honest scope:
    Color signature CAN help distinguish cars from DIFFERENT teams
    (e.g. a red car vs a blue car). It CANNOT distinguish teammates
    with near-identical livery (this was the exact case that broke
    BoT-SORT's appearance re-ID in Phase 4 - same problem, same
    limitation, for the same underlying reason). Don't expect this
    to solve the occlusion-swap problem on its own; it's a partial
    signal, reported with that caveat built into the output.

Usage:
    python src/identity.py --video data/videos/F1_FullVideo.mp4 \\
        --tracks-csv outputs/tracking/tracks_F1_FullVideo_from3000.csv \\
        --start-frame 3000

Output:
    outputs/analytics/track_colors_<video_name>.csv
        track_id, dominant_color_hex, sample_count, mean_bgr

    Also groups tracks into color clusters (simple distance
    threshold, not a full ML clustering step) and reports which
    track_ids likely share a similar livery - useful both as a
    sanity check and as a flag for "these tracks may be hard to
    tell apart even with this method."
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_OUT_DIR = PROJECT_ROOT / "outputs" / "analytics"


def load_tracks(csv_path: Path) -> dict:
    """Group CSV rows by frame index, so we can look up which boxes
    exist for a given frame while reading the video sequentially."""
    by_frame = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_frame[int(row["frame"])].append(row)
    return by_frame


def dominant_color(crop: np.ndarray) -> tuple:
    """
    True dominant color via k-means clustering, NOT a simple mean.

    Why this matters: F1 liveries are almost always multi-color (e.g.
    red+white, orange+black). Averaging two contrasting colors
    together produces a muddy, desaturated blend by definition - this
    is exactly what we saw (every track reporting a similar dark
    grey/brown regardless of actual livery, even on sharp, stationary
    grid-start footage where liveries are clearly distinct to the
    eye). A mean isn't measuring the "wrong" color, it's measuring a
    blend that doesn't correspond to any single part of the car.

    K-means with k=2 finds the two main color clusters in the crop
    (e.g. "red" and "white" for a Ferrari) and returns the LARGER
    cluster's center as the representative color - much closer to
    what a person means by "what color is this car."
    """
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    pixels = crop.reshape(-1, 3).astype(np.float32)

    # Drop near-black (tires/shadow) and near-white (glare) pixels
    # before clustering, so those don't dominate a cluster themselves.
    brightness = pixels.mean(axis=1)
    keep = (brightness > 30) & (brightness < 230)
    filtered = pixels[keep]

    if len(filtered) < 10:
        filtered = pixels  # not enough non-shadow/glare pixels, use all

    k = 2 if len(filtered) >= 2 else 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(filtered, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Pick the cluster with the most pixels - the actual dominant color
    counts = np.bincount(labels.flatten())
    dominant_idx = int(np.argmax(counts))
    dominant = centers[dominant_idx]

    return tuple(round(float(c), 1) for c in dominant)


def bgr_to_hex(bgr: tuple) -> str:
    b, g, r = bgr
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def extract_track_colors(video_path: Path, tracks_by_frame: dict, start_frame: int = 0,
                          save_crops: bool = False, crops_dir: Path = None) -> dict:
    """
    Walk the video once, and for every tracked box in every frame,
    sample its dominant color. Returns {track_id: [list of colors]}.

    save_crops: if True, also write the actual crop images to disk
    (a few per track) so results can be visually sanity-checked
    instead of trusted blindly from aggregate numbers alone.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    colors_by_track = defaultdict(list)
    crop_counts = defaultdict(int)
    idx = start_frame
    max_frame_needed = max(tracks_by_frame.keys()) if tracks_by_frame else start_frame

    if save_crops and crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    while idx <= max_frame_needed:
        ok, frame = cap.read()
        if not ok:
            break

        for row in tracks_by_frame.get(idx, []):
            x1, y1, x2, y2 = (int(float(row["x1"])), int(float(row["y1"])),
                               int(float(row["x2"])), int(float(row["y2"])))
            x1, y1 = max(x1, 0), max(y1, 0)
            crop = frame[y1:y2, x1:x2]
            color = dominant_color(crop)
            if color:
                colors_by_track[row["track_id"]].append(color)

            if save_crops and crops_dir and crop.size > 0 and crop_counts[row["track_id"]] < 3:
                track_dir = crops_dir / f"track_{row['track_id']}"
                track_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(track_dir / f"frame_{idx:06d}.jpg"), crop)
                crop_counts[row["track_id"]] += 1

        idx += 1

    cap.release()
    return colors_by_track


def summarize_colors(colors_by_track: dict) -> dict:
    results = {}
    for track_id, colors in colors_by_track.items():
        arr = np.array(colors)
        mean_bgr = tuple(round(float(c), 1) for c in arr.mean(axis=0))
        results[track_id] = {
            "mean_bgr": mean_bgr,
            "hex": bgr_to_hex(mean_bgr),
            "sample_count": len(colors),
        }
    return results


def group_similar_colors(summaries: dict, distance_threshold: float = 30.0) -> list:
    """
    Very simple grouping: two tracks are "similarly colored" if their
    mean BGR euclidean distance is below the threshold. Not a proper
    clustering algorithm - just enough to flag "these might be the
    same team / hard to distinguish" for manual review.
    """
    track_ids = list(summaries.keys())
    groups = []
    used = set()

    for i, tid_a in enumerate(track_ids):
        if tid_a in used:
            continue
        group = [tid_a]
        used.add(tid_a)
        color_a = np.array(summaries[tid_a]["mean_bgr"])

        for tid_b in track_ids[i + 1:]:
            if tid_b in used:
                continue
            color_b = np.array(summaries[tid_b]["mean_bgr"])
            distance = np.linalg.norm(color_a - color_b)
            if distance < distance_threshold:
                group.append(tid_b)
                used.add(tid_b)

        groups.append(group)

    return groups


def save_csv(summaries: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "dominant_color_hex", "mean_b", "mean_g", "mean_r", "sample_count"])
        for track_id, s in summaries.items():
            writer.writerow([track_id, s["hex"], *s["mean_bgr"], s["sample_count"]])
    print(f"Track color signatures saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Extract per-track livery color signature.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    parser.add_argument("--tracks-csv", type=str, required=True, help="Path to tracker.py's tracks CSV.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame the tracks CSV started at.")
    parser.add_argument("--save-crops", action="store_true",
                         help="Save a few actual crop images per track for visual sanity-checking.")
    args = parser.parse_args()

    video_path = Path(args.video)
    tracks_csv = Path(args.tracks_csv)
    if not video_path.exists() or not tracks_csv.exists():
        print("ERROR: video or tracks CSV not found.")
        sys.exit(1)

    tracks_by_frame = load_tracks(tracks_csv)
    print(f"Loaded tracks for {len(tracks_by_frame)} frame(s).")

    crops_dir = ANALYTICS_OUT_DIR / "crops" / video_path.stem if args.save_crops else None
    colors_by_track = extract_track_colors(video_path, tracks_by_frame, args.start_frame,
                                            args.save_crops, crops_dir)
    summaries = summarize_colors(colors_by_track)

    print(f"\n{'=' * 60}")
    print("TRACK COLOR SIGNATURES")
    print(f"{'=' * 60}")
    for tid, s in summaries.items():
        print(f"  track_id {tid}: {s['hex']}  (from {s['sample_count']} sample(s))")

    groups = group_similar_colors(summaries)
    multi_groups = [g for g in groups if len(g) > 1]
    print(f"\nSimilarly-colored track groups (possible same team, or same car re-IDed):")
    if multi_groups:
        for g in multi_groups:
            print(f"  {g}")
        print("\n  NOTE: tracks grouped here may be the SAME physical car whose track_id")
        print("  reset (Phase 4 dropout), OR two genuinely different cars from the same")
        print("  team with similar livery. This script cannot tell those apart on its own -")
        print("  it only flags the ambiguity for manual review or a further signal")
        print("  (e.g. position continuity) to resolve.")
    else:
        print("  None - every track has a visually distinct color signature.")

    out_path = ANALYTICS_OUT_DIR / f"track_colors_{video_path.stem}.csv"
    save_csv(summaries, out_path)


if __name__ == "__main__":
    main()