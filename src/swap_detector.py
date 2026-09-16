"""
src/swap_detector.py

Purpose
-------
We found direct evidence (track_id 3's speed spiking to 465 px/s
against its own 36 px/s average - ~13x) that occlusion swaps leave a
detectable fingerprint: an implausible speed jump that doesn't match
the car's own recent motion.

This script doesn't PREVENT swaps (that needs Phase 7-level motion
modeling), but it DETECTS them from data we already have, and stops
them from silently corrupting a track: when a suspicious jump is
found, the track_id is split into two new IDs at that frame, so nothing
downstream (position analysis, future phases) treats the pre-swap and
post-swap segments as one continuous, trustworthy path.

Method: for each track, compute a rolling average speed from the
last few frames, then flag any frame where the instantaneous speed
exceeds that average by more than `threshold_ratio`. This adapts to
each car's own typical speed, rather than using one global cutoff
(a car doing 100px/s steadily should not be flagged; a car jumping
from 10px/s to 100px/s in one frame should be).

Usage:
    python src/swap_detector.py --trajectories-csv outputs/analytics/trajectories_tracks_F1_FullVideo.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_OUT_DIR = PROJECT_ROOT / "outputs" / "analytics"


def load_rows(csv_path: Path) -> list:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def detect_and_split(rows: list, threshold_ratio: float = 4.0, window: int = 5) -> list:
    """
    For each track_id (sorted by frame), walk through its speed
    values. Maintain a rolling average of the last `window` speeds.
    If the current speed exceeds rolling_avg * threshold_ratio (and
    rolling_avg is non-trivial), flag this frame as a likely swap
    point and start a new track_id from here onward.

    Returns a new list of rows with an added 'split_track_id' column -
    the original track_id, or a suffixed version (_b, _c, ...) after
    each detected split.
    """
    by_track = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)
    for tid in by_track:
        by_track[tid].sort(key=lambda r: int(r["frame"]))

    output_rows = []
    split_log = []

    for tid, track_rows in by_track.items():
        current_id = tid
        split_count = 0
        recent_speeds = []

        for row in track_rows:
            speed_str = row["pixel_speed_px_per_s"]
            speed = float(speed_str) if speed_str not in ("", None) else None

            if speed is not None and recent_speeds:
                rolling_avg = sum(recent_speeds) / len(recent_speeds)
                if rolling_avg > 1.0 and speed > rolling_avg * threshold_ratio:
                    split_count += 1
                    current_id = f"{tid}_split{split_count}"
                    split_log.append({
                        "original_track_id": tid,
                        "split_at_frame": row["frame"],
                        "speed_before": round(rolling_avg, 1),
                        "speed_spike": round(speed, 1),
                        "new_track_id": current_id,
                    })
                    recent_speeds = []  # reset rolling window for the new segment

            new_row = dict(row)
            new_row["split_track_id"] = current_id
            output_rows.append(new_row)

            if speed is not None:
                recent_speeds.append(speed)
                if len(recent_speeds) > window:
                    recent_speeds.pop(0)

    return output_rows, split_log


def save_csv(rows: list, out_path: Path):
    if not rows:
        print("No rows to save.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Split-corrected trajectories saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect likely occlusion swaps via speed-spike analysis.")
    parser.add_argument("--trajectories-csv", type=str, required=True,
                         help="Path to a trajectories_*.csv file from position_movement.py.")
    parser.add_argument("--threshold-ratio", type=float, default=4.0,
                         help="Flag a frame if speed exceeds this multiple of the recent rolling average.")
    args = parser.parse_args()

    csv_path = Path(args.trajectories_csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist.")
        sys.exit(1)

    rows = load_rows(csv_path)
    output_rows, split_log = detect_and_split(rows, args.threshold_ratio)

    print(f"\n{'=' * 60}")
    print("SWAP DETECTION RESULTS")
    print(f"{'=' * 60}")
    if not split_log:
        print("  No suspicious speed spikes found - no splits made.")
    else:
        for entry in split_log:
            print(f"  track_id {entry['original_track_id']}: suspected swap at frame "
                  f"{entry['split_at_frame']} (speed jumped {entry['speed_before']} -> "
                  f"{entry['speed_spike']} px/s) -> split into '{entry['new_track_id']}'")

    out_path = ANALYTICS_OUT_DIR / f"{csv_path.stem}_swap_corrected.csv"
    save_csv(output_rows, out_path)

    print(f"\nUse 'split_track_id' (not 'track_id') downstream to avoid trusting")
    print(f"continuity across a detected swap point.")


if __name__ == "__main__":
    main()