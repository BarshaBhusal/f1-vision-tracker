"""
src/tracker.py

Phase 4 - Multi-Object Tracking
---------------------------------
Purpose:
    Detection alone gives us boxes per frame with no memory between
    frames. This phase adds persistent identity: the same physical
    car should keep the same track_id across consecutive frames, even
    as it moves, gets briefly occluded, etc.

Why ByteTrack (not BoT-SORT/DeepSORT):
    ByteTrack is motion/IoU-based (no appearance re-identification
    model), which makes it lightweight enough for CPU-only operation
    (see Phase 3 hardware discussion - AMD integrated GPU, no CUDA/
    ROCm path). BoT-SORT adds appearance-feature re-ID on top, which
    costs more compute for a benefit we don't need yet (all our cars
    are visually similar - team livery, not persistent facial-recog-
    style features - so cheap appearance matching wouldn't help much
    anyway at this stage). If ID switches turn out to be a major
    problem in testing, BoT-SORT is the documented next thing to try.

Why consecutive frames, not sampled frames:
    Phase 3 testing intentionally sampled sparse, spread-out frames
    to get a diverse look at detection quality. Tracking is the
    opposite: it depends on frame-to-frame continuity to associate
    detections into tracks. Skipping frames breaks that continuity
    and will fragment tracks or cause ID switches that have nothing
    to do with the tracker's actual quality. So this script processes
    frames in sequence, with --max-frames as a way to limit total
    RUNTIME (useful for CPU smoke-testing) without skipping frames
    within that range.

Settings locked in from Phase 3 testing:
    classes = car, truck   (car alone missed a pit-lane misclassification)
    iou     = 0.4          (reduced duplicate/overlapping boxes on
                             packed grid cars, at the cost of one
                             ambiguous case worth re-checking here)

Usage:
    # Smoke test: track the first 150 frames only (~3 sec at ~50fps)
    python src/tracker.py --video data/videos/F1_FullVideo.mp4 --max-frames 150

    # Full video (slow on CPU - expect this to take a while)
    python src/tracker.py --video data/videos/F1_FullVideo.mp4

Output:
    outputs/annotated_videos/<video_name>_tracked.mp4   - boxes + track IDs
    outputs/tracking/tracks_<video_name>.csv             - structured track data
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATED_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "annotated_videos"
TRACKING_OUT_DIR = PROJECT_ROOT / "outputs" / "tracking"
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_CLASSES = ["car", "truck"]  # locked in from Phase 3 findings
DEFAULT_IOU = 0.4  # locked in from Phase 3 findings

# F1 cars sit on the boundary between COCO's "car" and "truck" classes -
# Phase 3 found this wasn't rare noise (track_analysis.py measured 5/11
# tracks flip-flopping between the two, some near 50/50 across a single
# track). Neither label is more "correct" than the other for this
# object, so instead of picking a winner after the fact, we collapse
# both into one canonical class at the source. Downstream phases
# (identity, position, laps, analytics) should never have to treat
# "car" vs "truck" as if they meant different things.
CANONICAL_CLASS = "f1_car"
CANONICALIZE = {"car": CANONICAL_CLASS, "truck": CANONICAL_CLASS}


def get_class_ids(model: YOLO, class_names: list) -> list:
    return [k for k, v in model.names.items() if v in class_names]


def canonical_name(raw_name: str) -> str:
    """Map COCO's ambiguous car/truck split onto one canonical label."""
    return CANONICALIZE.get(raw_name, raw_name)


def run_tracking(video_path: Path, model: YOLO, class_ids: list, conf: float,
                  iou: float, tracker_cfg: str, max_frames: int = None, start_frame: int = 0) -> tuple:
    """
    Run tracking frame-by-frame over the video using model.track()
    with persist=True, which tells Ultralytics to maintain track
    state across successive calls (rather than treating each frame
    as an independent detection problem).

    start_frame: skip to this frame index before tracking begins.
    Note this means the tracker starts "cold" at that point (no prior
    track history), which is fine for stress-testing a segment in
    isolation but means track IDs won't carry over from frame 0.

    Returns (rows, output_video_path).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        if start_frame >= total_frames:
            raise ValueError(f"--start-frame {start_frame} is beyond total frame count {total_frames}.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_available = total_frames - start_frame
    frame_limit = min(max_frames, frames_available) if max_frames else frames_available

    ANNOTATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_from{start_frame}" if start_frame > 0 else ""
    out_video_path = ANNOTATED_VIDEOS_DIR / f"{video_path.stem}_tracked{suffix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    rows = []
    idx = 0
    seen_ids = set()

    print(f"Tracking {frame_limit} frame(s) of {video_path.name}, "
          f"starting at frame {start_frame} ...")

    while idx < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            persist=True,
            conf=conf,
            iou=iou,
            classes=class_ids,
            tracker=tracker_cfg,
            device="cpu",
            verbose=False,
        )

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        annotated = frame.copy()

        boxes = results[0].boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                tid = int(track_id)
                seen_ids.add(tid)

                raw_class = model.names[cls_id]
                stable_class = canonical_name(raw_class)

                rows.append({
                    "frame": start_frame + idx,
                    "timestamp_ms": round(timestamp_ms, 1),
                    "track_id": tid,
                    "class_name": stable_class,
                    "raw_class_name": raw_class,
                    "confidence": round(conf_score, 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "center_x": round((x1 + x2) / 2, 1),
                    "center_y": round((y1 + y2) / 2, 1),
                })

                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                label = f"ID {tid} {stable_class} {conf_score:.2f}"
                cv2.putText(annotated, label, (p1[0], max(p1[1] - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        writer.write(annotated)

        if idx % 50 == 0:
            print(f"  frame {start_frame + idx} ({idx}/{frame_limit} in this run) - "
                  f"{len(seen_ids)} unique track ID(s) so far")

        idx += 1

    cap.release()
    writer.release()

    print(f"\nDone: {idx} frame(s) processed, {len(seen_ids)} unique track ID(s) total.")
    print(f"Annotated video: {out_video_path}")

    return rows, out_video_path


def save_tracks_csv(rows: list, csv_path: Path):
    if not rows:
        print("No tracking data to save.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Tracks CSV saved to: {csv_path}")


def summarize_tracks(rows: list):
    """
    Quick diagnostic: how many frames does each track_id appear in?
    Very short-lived tracks (e.g. 1-3 frames) are a red flag for
    fragmentation or false-positive detections rather than real cars.
    """
    from collections import Counter
    counts = Counter(r["track_id"] for r in rows)
    short_lived = {tid: n for tid, n in counts.items() if n <= 3}

    print(f"\nTrack length summary ({len(counts)} unique IDs):")
    print(f"  Longest-lived track: ID {max(counts, key=counts.get)} "
          f"({max(counts.values())} frames)")
    print(f"  Short-lived tracks (<=3 frames, possible fragmentation/noise): {len(short_lived)}")
    if short_lived:
        print(f"    IDs: {sorted(short_lived.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Multi-object tracking with ByteTrack.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES),
                         help="Comma-separated COCO class names to track.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         help="Ultralytics tracker config: 'bytetrack.yaml', 'botsort.yaml', or a path to a custom tracker YAML.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model weights.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Limit total frames processed (for CPU smoke-testing).")
    parser.add_argument("--start-frame", type=int, default=0,
                         help="Skip to this frame index before tracking begins (tracker starts cold here, no history from frame 0).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: {video_path} does not exist.")
        sys.exit(1)

    classes = [c.strip() for c in args.classes.split(",")]
    print(f"Loading model: {args.model} (CPU)")
    model = YOLO(args.model)
    class_ids = get_class_ids(model, classes)

    rows, _ = run_tracking(video_path, model, class_ids, args.conf, args.iou,
                            args.tracker, args.max_frames, args.start_frame)

    csv_suffix = f"_from{args.start_frame}" if args.start_frame > 0 else ""
    save_tracks_csv(rows, TRACKING_OUT_DIR / f"tracks_{video_path.stem}{csv_suffix}.csv")
    summarize_tracks(rows)


if __name__ == "__main__":
    main()