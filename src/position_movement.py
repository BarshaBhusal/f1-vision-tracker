"""
src/position_movement.py

Phase 6 - Position and Movement
----------------------------------
Purpose:
    Turn per-frame bounding boxes into per-car TRAJECTORIES: how does
    each track_id move through the frame over time? This is the
    first phase that treats a car's path as a continuous thing, not
    just a series of independent boxes.

Critical honesty constraint (Section 15 of the spec):
    Pixel movement is NOT real-world speed. A car's on-screen speed
    changes with camera zoom, panning, and perspective - a car
    "moving 40px/frame" in a wide shot and a close-up onboard shot
    are not comparable, and neither corresponds to a real km/h
    number without knowing the camera's calibration (which we don't
    have yet - that's Phase 7, track/camera understanding).

    So this script reports:
        DIRECTLY OBSERVED: pixel position, pixel displacement,
        pixel-per-second rate, direction of travel in image space.

        NOT PROVIDED: real-world speed, real-world distance, lap
        time contribution - these require Phase 7's camera
        calibration and would be fabricated numbers without it.

Usage:
    python src/position_movement.py --tracks-csv outputs/tracking/tracks_F1_FullVideo.csv \\
        --video data/videos/F1_FullVideo.mp4

Output:
    outputs/analytics/trajectories_<video>.csv
        frame, timestamp_ms, track_id, center_x, center_y,
        dx_px, dy_px, pixel_speed_px_per_s, direction_deg

    outputs/analytics/trajectory_overlay_<video>.jpg
        visual plot of every track's path, drawn on a representative
        frame, so trajectories can be sanity-checked by eye
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_OUT_DIR = PROJECT_ROOT / "outputs" / "analytics"

TRACK_COLORS = [
    (255, 99, 71), (60, 179, 113), (30, 144, 255), (255, 215, 0),
    (238, 130, 238), (0, 255, 255), (255, 140, 0), (154, 205, 50),
    (199, 21, 133), (70, 130, 180), (255, 20, 147), (0, 250, 154),
]


def color_for_track(track_id: str) -> tuple:
    return TRACK_COLORS[hash(track_id) % len(TRACK_COLORS)]


def load_rows(csv_path: Path) -> list:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def compute_trajectories(rows: list) -> dict:
    """
    Group rows by track_id (sorted by frame), then compute
    frame-to-frame pixel displacement, speed, and direction.

    Returns {track_id: [enriched row dicts]}.
    """
    by_track = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)

    for tid in by_track:
        by_track[tid].sort(key=lambda r: int(r["frame"]))

    trajectories = {}
    for tid, track_rows in by_track.items():
        enriched = []
        prev = None
        for row in track_rows:
            cx, cy = float(row["center_x"]), float(row["center_y"])
            t_ms = float(row["timestamp_ms"])

            dx = dy = pixel_speed = direction_deg = None
            if prev is not None:
                dx = cx - prev["cx"]
                dy = cy - prev["cy"]
                dt_s = (t_ms - prev["t_ms"]) / 1000.0
                dist = math.hypot(dx, dy)
                pixel_speed = round(dist / dt_s, 2) if dt_s > 0 else None
                direction_deg = round(math.degrees(math.atan2(dy, dx)), 1)

            enriched.append({
                "frame": row["frame"],
                "timestamp_ms": row["timestamp_ms"],
                "track_id": tid,
                "center_x": round(cx, 1),
                "center_y": round(cy, 1),
                "dx_px": round(dx, 2) if dx is not None else "",
                "dy_px": round(dy, 2) if dy is not None else "",
                "pixel_speed_px_per_s": pixel_speed if pixel_speed is not None else "",
                "direction_deg": direction_deg if direction_deg is not None else "",
            })
            prev = {"cx": cx, "cy": cy, "t_ms": t_ms}

        trajectories[tid] = enriched

    return trajectories


def save_trajectories_csv(trajectories: dict, out_path: Path):
    all_rows = [row for rows in trajectories.values() for row in rows]
    all_rows.sort(key=lambda r: (int(r["frame"]), r["track_id"]))

    if not all_rows:
        print("No trajectory data to save.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Trajectories CSV saved to: {out_path}")


def draw_trajectory_overlay(video_path: Path, trajectories: dict, out_path: Path):
    """
    Draw every track's path as a colored line on top of a single
    representative frame (the last frame with data), so trajectories
    can be visually sanity-checked rather than trusted blindly.
    """
    all_frames = sorted({int(r["frame"]) for rows in trajectories.values() for r in rows})
    if not all_frames:
        return
    bg_frame_idx = all_frames[-1]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARNING: could not open {video_path} for overlay background.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("WARNING: could not read background frame for overlay.")
        return

    overlay = frame.copy()
    for tid, rows in trajectories.items():
        color = color_for_track(tid)
        points = [(int(float(r["center_x"])), int(float(r["center_y"]))) for r in rows]
        for i in range(1, len(points)):
            cv2.line(overlay, points[i - 1], points[i], color, 2)
        if points:
            cv2.circle(overlay, points[-1], 4, color, -1)
            cv2.putText(overlay, tid, points[-1], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"Trajectory overlay saved to: {out_path}")


def print_summary(trajectories: dict):
    print(f"\n{'=' * 60}")
    print("TRAJECTORY SUMMARY (pixel-space, NOT real-world speed)")
    print(f"{'=' * 60}")
    for tid, rows in trajectories.items():
        speeds = [r["pixel_speed_px_per_s"] for r in rows if r["pixel_speed_px_per_s"] != ""]
        if not speeds:
            print(f"  track_id {tid}: only 1 frame, no movement data")
            continue
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        print(f"  track_id {tid}: {len(rows)} frames, "
              f"avg {avg_speed:.1f} px/s, max {max_speed:.1f} px/s")
    print(f"\nNOTE: px/s values reflect ON-SCREEN motion only. Camera panning,")
    print(f"zoom, and perspective are NOT accounted for - these are NOT real-world")
    print(f"speeds. Real-world speed requires camera calibration (Phase 7).")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Compute per-track trajectories and pixel-space movement.")
    parser.add_argument("--tracks-csv", type=str, required=True, help="Path to tracker.py's tracks CSV.")
    parser.add_argument("--video", type=str, required=True, help="Video file, used only for the trajectory overlay image.")
    args = parser.parse_args()

    tracks_csv = Path(args.tracks_csv)
    video_path = Path(args.video)
    if not tracks_csv.exists():
        print(f"ERROR: {tracks_csv} does not exist.")
        sys.exit(1)

    rows = load_rows(tracks_csv)
    trajectories = compute_trajectories(rows)

    save_trajectories_csv(trajectories, ANALYTICS_OUT_DIR / f"trajectories_{tracks_csv.stem}.csv")
    draw_trajectory_overlay(video_path, trajectories, ANALYTICS_OUT_DIR / f"trajectory_overlay_{tracks_csv.stem}.jpg")
    print_summary(trajectories)


if __name__ == "__main__":
    main()