import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version

import yaml
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from .history import Trails
from .color_assist import ColorAssist

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description="YOLO + built-in ByteTrack sports ball tracker")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    p.add_argument("--color-assist", action="store_true", help="Optional YOLO-seeded color observations for a single distinct-color ball")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--device", default="cpu", help="cpu / 0 / cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--tracker", type=Path, default=ROOT / "configs/bytetrack.yaml")
    p.add_argument("--output", type=Path, default=Path("runs/demo"))
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--trail", type=int, default=90)
    p.add_argument("--threads", type=int, default=4)
    return p


def run(args):
    if not 0 < args.conf <= 1 or args.imgsz < 32 or args.trail < 1 or args.threads < 1 or args.max_frames < 0:
        raise ValueError("Invalid confidence, image size, trail, thread or frame limit")
    if args.video and not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output folder is not empty; use a new --output folder")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    model = YOLO(args.model)
    names = model.names
    classes = [i for i, name in names.items() if name == "sports ball"]
    if not classes:
        raise ValueError("Model must contain the COCO sports ball class")
    assistant = ColorAssist(classes[0]) if args.color_assist else None
    if assistant is not None:
        model.add_callback("on_predict_postprocess_end", assistant)
    cap = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
    writer = None
    frames, infer_seconds, processing_seconds = 0, 0.0, 0.0
    detected_frames = 0
    ids = set()
    tracker_config = yaml.safe_load(args.tracker.read_text(encoding="utf-8"))
    trails = Trails(args.trail, max_gap=int(tracker_config["track_buffer"]))
    csv_file = None
    timing_file = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video/camera. Check path, camera permissions or device index.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and 1 <= fps <= 240 else 30.0
        csv_file = (args.output / "tracks.csv").open("w", newline="", encoding="utf-8")
        fields = ["frame", "time_s", "id", "x1", "y1", "x2", "y2", "confidence"]
        table = csv.DictWriter(csv_file, fieldnames=fields)
        table.writeheader()
        timing_file = (args.output / "frames.csv").open("w", newline="", encoding="utf-8")
        timings = csv.writer(timing_file)
        timings.writerow(["frame", "inference_s", "processing_s", "visible_tracks"])
        started = time.perf_counter()
        while not args.max_frames or frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            tick = time.perf_counter()
            result = model.track(frame, persist=True, tracker=str(args.tracker), classes=classes,
                                 conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            infer_duration = time.perf_counter() - tick
            infer_seconds += infer_duration
            observations = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                detected_frames += 1
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
                for box, confidence, tid in zip(xyxy, confs, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1+x2)//2, (y1+y2)//2)
                    color = (0, 210, 255) if tid < 0 else ((tid*67)%180+60, (tid*97)%180+60, (tid*43)%180+60)
                    if tid >= 0:
                        ids.add(tid)
                        observations.append((tid, center))
                    table.writerow(dict(zip(fields, [frames, frames/fps, tid, *map(float, box), float(confidence)])))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, center, 3, color, -1)
                    cv2.putText(frame, f"ID:{tid if tid >= 0 else '?'} ball {confidence:.2f}",
                                (max(0,x1), max(18,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            trails.update(frames, observations)
            for tid, _ in observations:
                for a, b in trails.segments(tid):
                    cv2.line(frame, a, b, (255,0,255), 3)
            # Running processing rate excludes capture, display waits, and startup; includes inference/render/write.
            current_fps = (frames+1) / max(processing_seconds + time.perf_counter()-tick, 1e-9)
            cv2.putText(frame, f"FPS:{current_fps:.1f} visible:{len(observations)}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(args.output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 writer unavailable; check OpenCV codec and output path")
                writer.write(frame)
            duration = time.perf_counter() - tick
            processing_seconds += duration
            timings.writerow([frames, infer_duration, duration, len(observations)])
            frames += 1
            if not args.no_display:
                cv2.imshow("Ball Tracking - Q/ESC to quit", frame)
                if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                    break
        if frames == 0:
            raise RuntimeError("Input opened but no frames decoded")
        elapsed = time.perf_counter()-started
        model_path = Path(args.model)
        summary = dict(frames=frames, frames_with_detections=detected_frames,
                       detection_frame_fraction=detected_frames/frames, unique_track_ids=len(ids),
                       inference_fps=frames/infer_seconds, processing_fps=frames/processing_seconds,
                       wall_fps=frames/elapsed, source_fps=fps, python=platform.python_version(),
                       ultralytics=ultralytics.__version__, torch=torch.__version__, opencv=cv2.__version__,
                       platform=platform.platform(), processor=platform.processor(), threads=args.threads,
                       packages={p:version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "scipy", "lap")},
                       device=args.device, args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                       tracker_yaml=args.tracker.read_text(encoding="utf-8"),
                       model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None)
        summary["color_assist_inserted_observations"] = assistant.inserted if assistant else 0
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if timing_file is not None:
            timing_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
