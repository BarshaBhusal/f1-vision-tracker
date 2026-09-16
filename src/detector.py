"""
src/detector.py

Phase 3 - Car Detection
------------------------
Purpose:
    Run a pretrained YOLOv8 model over sampled frames to establish a
    BASELINE before deciding anything about fine-tuning. We are not
    assuming a generic COCO-trained model will correctly recognize
    F1 cars as "car" - it might call them "truck", split one car into
    two detections, miss small/packed cars entirely, or false-positive
    on trackside objects. This script exists to SHOW us that behavior,
    not to assume it works.

Why no class filtering by default:
    COCO's "car" class (id 2) was the obvious guess, but F1 cars are
    visually unusual (low, wide, wings, exposed wheels) compared to
    COCO's training distribution of road cars. Filtering to only
    "car" before we've looked at results risks silently discarding
    correct detections that landed under "truck" or another class.
    So by default this script detects across ALL COCO classes and
    reports what it found - you decide the class filter afterward,
    once we've seen the real behavior on real frames.

Two input modes:
    1. --images-dir   Run on already-saved JPGs (e.g. from
                       outputs/frames/F1_FullVideo/) - fastest way to
                       eyeball a handful of frames first, no video
                       decoding needed.
    2. --video         Sample N frames directly from a video via
                       frame_extractor.iter_frames() and run on those.

CPU-only by design (see hardware discussion - AMD integrated GPU,
no usable CUDA/ROCm path on this machine). Model defaults to
YOLOv8s as the accuracy/speed tradeoff for small, packed objects.

Usage:
    # Baseline test on a handful of already-saved frames
    python src/detector.py --images-dir outputs/frames/F1_FullVideo --limit 8

    # Sample 10 frames directly from a video
    python src/detector.py --video data/videos/F1_FullVideo.mp4 --count 10

    # Once we've seen results, restrict to specific classes
    python src/detector.py --images-dir outputs/frames/F1_FullVideo --classes car,truck
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

# frame_extractor.py must be in the same src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_extractor import iter_frames  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTIONS_OUT_DIR = PROJECT_ROOT / "outputs" / "detections"
DEFAULT_MODEL = "yolov8s.pt"  # auto-downloaded by ultralytics on first run


def load_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    print(f"Loading model: {model_name} (CPU)")
    model = YOLO(model_name)
    return model


def run_detection(model: YOLO, frame, conf: float, classes: list = None, iou: float = 0.7,
                   upscale: float = 1.0) -> list:
    """
    Run YOLO on a single frame.

    Returns a list of dicts: class_name, confidence, x1, y1, x2, y2,
    center_x, center_y - the fields we'll want for Phase 4 tracking
    and the detections.csv output per the spec's Section 20 format.

    iou: Non-Max Suppression IoU threshold. Lower = more aggressive at
    merging/suppressing overlapping boxes (helps with duplicate boxes
    on tightly-packed cars, but risks merging two genuinely distinct
    adjacent cars into one). Default 0.7 is YOLO's stock value; we
    observed duplicate/overlapping boxes on packed grid-start cars at
    this setting, so worth testing lower (e.g. 0.4-0.5).

    upscale: resize the frame by this factor BEFORE running YOLO.
    Our source footage is low-resolution (848x384), and small/distant
    cars get very few actual pixels - this was traced back as the
    likely root cause behind multiple separate problems (weak
    detection confidence on distant cars, unusable identity crops,
    muddy color signatures). Upscaling gives the model more pixels to
    work with, at the cost of slower inference (bigger image = more
    compute). Boxes are rescaled back to ORIGINAL frame coordinates
    before being returned, so nothing downstream (tracker.py,
    identity.py) needs to know this happened.
    """
    class_ids = None
    if classes:
        class_ids = [k for k, v in model.names.items() if v in classes]

    infer_frame = frame
    if upscale != 1.0:
        infer_frame = cv2.resize(frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    results = model.predict(
        source=infer_frame,
        conf=conf,
        iou=iou,
        classes=class_ids,
        device="cpu",
        verbose=False,
    )

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if upscale != 1.0:
            x1, y1, x2, y2 = x1 / upscale, y1 / upscale, x2 / upscale, y2 / upscale
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "center_x": round((x1 + x2) / 2, 1),
            "center_y": round((y1 + y2) / 2, 1),
        })
    return detections


def draw_detections(frame, detections: list):
    """Draw boxes + class/confidence labels for visual inspection."""
    annotated = frame.copy()
    for det in detections:
        p1 = (int(det["x1"]), int(det["y1"]))
        p2 = (int(det["x2"]), int(det["y2"]))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.putText(annotated, label, (p1[0], max(p1[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return annotated


def process_images_dir(model, images_dir: Path, conf: float, classes: list, limit: int, iou: float = 0.7,
                        tag: str = None, upscale: float = 1.0) -> tuple:
    """Run detection over saved JPGs in a directory. Returns (rows, out_dir)."""
    image_paths = sorted(images_dir.glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        print(f"ERROR: no .jpg files found in {images_dir}")
        sys.exit(1)

    folder_name = f"{images_dir.name}_{tag}" if tag else images_dir.name
    out_dir = DETECTIONS_OUT_DIR / "annotated" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path}, skipping.")
            continue
        detections = run_detection(model, frame, conf, classes, iou, upscale)
        annotated = draw_detections(frame, detections)
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        print(f"  {img_path.name}: {len(detections)} detection(s) "
              f"[{', '.join(d['class_name'] for d in detections) or 'none'}]")

        for det in detections:
            rows.append({"source": img_path.name, **det})

    print(f"\nAnnotated images saved to: {out_dir}")
    return rows, out_dir


def process_video(model, video_path: Path, conf: float, classes: list, count: int) -> tuple:
    """Sample N frames from a video and run detection on each."""
    out_dir = DETECTIONS_OUT_DIR / "annotated" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Determine target frame indices (evenly spaced), same approach as
    # frame_extractor.extract_frames_to_disk, without saving raw frames.
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        print(f"ERROR: could not read frame count from {video_path}")
        sys.exit(1)

    n = max(1, min(count, total_frames))
    target_indices = set(int(total_frames * (i + 1) / (n + 1)) for i in range(n))

    for fd in iter_frames(video_path, frame_skip=1):
        if fd.index not in target_indices:
            if fd.index >= max(target_indices):
                break
            continue

        detections = run_detection(model, fd.frame, conf, classes)
        annotated = draw_detections(fd.frame, detections)
        out_path = out_dir / f"frame_{fd.index:06d}.jpg"
        cv2.imwrite(str(out_path), annotated)

        print(f"  frame {fd.index}: {len(detections)} detection(s) "
              f"[{', '.join(d['class_name'] for d in detections) or 'none'}]")

        for det in detections:
            rows.append({"source": f"frame_{fd.index:06d}", "frame_index": fd.index,
                         "timestamp_ms": round(fd.timestamp_ms, 1), **det})

        if fd.index >= max(target_indices):
            break

    print(f"\nAnnotated frames saved to: {out_dir}")
    return rows, out_dir


def save_csv(rows: list, csv_path: Path):
    if not rows:
        print("No detections to save to CSV.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detections CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Run YOLOv8 detection on sampled frames.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of saved JPG frames.")
    parser.add_argument("--video", type=str, default=None, help="Video file to sample frames from.")
    parser.add_argument("--count", type=int, default=10, help="Frames to sample when using --video.")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process when using --images-dir.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated COCO class names to filter to (default: all classes).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model weights to use.")
    parser.add_argument("--iou", type=float, default=0.7,
                         help="NMS IoU threshold. Lower (e.g. 0.4-0.5) reduces duplicate/overlapping boxes on packed objects.")
    parser.add_argument("--tag", type=str, default=None,
                         help="Optional label appended to the output folder name, so different parameter runs don't overwrite each other (e.g. --tag iou07).")
    parser.add_argument("--upscale", type=float, default=1.0,
                         help="Resize factor applied before detection (e.g. 2.0 = double resolution). Helps small/distant objects at the cost of speed.")
    args = parser.parse_args()

    if not args.images_dir and not args.video:
        print("ERROR: provide --images-dir or --video.")
        sys.exit(1)

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    model = load_model(args.model)

    if args.images_dir:
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            print(f"ERROR: {images_dir} does not exist.")
            sys.exit(1)
        rows, out_dir = process_images_dir(model, images_dir, args.conf, classes, args.limit, args.iou, args.tag, args.upscale)
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: {video_path} does not exist.")
            sys.exit(1)
        rows, out_dir = process_video(model, video_path, args.conf, classes, args.count)

    save_csv(rows, DETECTIONS_OUT_DIR / (f"detections_{args.tag}.csv" if args.tag else "detections.csv"))

    # Quick class-distribution summary - the actual diagnostic we're after
    from collections import Counter
    class_counts = Counter(r["class_name"] for r in rows)
    print("\nClass distribution across all detections:")
    if class_counts:
        for cls, n in class_counts.most_common():
            print(f"  {cls}: {n}")
    else:
        print("  (no detections at all - check conf threshold / model / input)")


if __name__ == "__main__":
    main()