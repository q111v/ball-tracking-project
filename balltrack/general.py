"""Official COCO sports-ball detection and Ultralytics built-in ByteTrack."""

import csv
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from .display_trail import DisplayTrail
from .motion import CameraMotion
from .runtime import configure_model

ROOT = Path(__file__).resolve().parents[1]


def sports_ball_class(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for identity, name in items:
        if name.lower().replace("_", " ") == "sports ball":
            return int(identity)
    raise ValueError(
        "This model has no COCO sports ball class; use official COCO weights."
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def process_video(
    video,
    output,
    *,
    model_path=None,
    device="cpu",
    image_size=960,
    confidence=0.1,
    camera_compensation=True,
    trail_seconds=0.7,
    trail_smoothing=0.8,
    save=True,
    show_window=True,
    threads=2,
    max_frames=0,
):
    video, output = Path(video), Path(output)
    model_path = Path(model_path or ROOT / "models/yolov8n.pt")
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0 < confidence <= 1 or image_size < 32 or threads < 1 or max_frames < 0:
        raise ValueError("Invalid confidence, image size, threads or frame limit")
    if not 0 < trail_seconds <= 10 or not 0 < trail_smoothing <= 1:
        raise ValueError("Invalid trail settings")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    model = configure_model(YOLO(str(model_path)), threads)
    ball_class = sports_ball_class(model.names)
    camera, trails, states = CameraMotion(camera_compensation), {}, Counter()
    cap, writer, table_file = cv2.VideoCapture(str(video)), None, None
    frames = observed_frames = tracked_frames = tracked_boxes = 0
    identities = set()
    # IDs may restart after cuts; (shot, id) identifies a track in exported data.
    shot = 0
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 30.0
        table_file = (output / "tracks.csv").open("w", newline="", encoding="utf-8")
        table = csv.writer(table_file)
        table.writerow(
            [
                "frame",
                "time_s",
                "shot",
                "id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "state",
            ]
        )
        while not max_frames or frames < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            matrix, motion_state = camera.update(image)
            states[motion_state] += 1
            if motion_state == "cut":
                shot += 1
                for tracker in getattr(model.predictor, "trackers", []):
                    tracker.reset()
                trails.clear()
            result = model.track(
                image,
                persist=True,
                tracker=str(ROOT / "configs/bytetrack.yaml"),
                classes=[ball_class],
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
            )[0]
            boxes = result.boxes
            active = set()
            has_detection = boxes is not None and len(boxes) > 0
            has_track = False
            if has_detection:
                observed_frames += 1
                coordinates = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.is_track
                    else [-1] * len(boxes)
                )
                for box, score, identity in zip(coordinates, scores, ids):
                    center = (box[:2] + box[2:]) / 2
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(image, tuple(center.astype(int)), 3, (0, 0, 255), -1)
                    label = f'Ball ID:{identity if identity>=0 else "?"} {score:.2f}'
                    cv2.putText(
                        image,
                        label,
                        (max(0, x1 - 20), max(50, y1 - 8)),
                        0,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    table.writerow(
                        [
                            frames,
                            frames / fps,
                            shot,
                            identity,
                            *map(float, box),
                            float(score),
                            "tracked" if identity >= 0 else "unconfirmed",
                        ]
                    )
                    if identity < 0:
                        continue
                    has_track = True
                    tracked_boxes += 1
                    identities.add((shot, identity))
                    active.add(identity)
                    trail = trails.setdefault(
                        identity, DisplayTrail(trail_seconds, trail_smoothing)
                    )
                    trail.update(
                        frames,
                        fps,
                        matrix,
                        identity,
                        center,
                        max(box[2] - box[0], box[3] - box[1]),
                    )
            else:
                table.writerow(
                    [frames, frames / fps, shot, -1, 0, 0, 0, 0, 0, "missing"]
                )
            tracked_frames += int(has_track)
            for identity in list(trails):
                trail = trails[identity]
                if identity not in active:
                    trail.update(frames, fps, matrix, -1, None)
                trail.draw(image, frames, fps)
                if not trail.points:
                    del trails[identity]
            rate = (frames + 1) / max(time.perf_counter() - started, 1e-9)
            cv2.putText(
                image,
                f"COCO sports ball | ByteTrack | FPS:{rate:.1f}",
                (15, 28),
                0,
                0.6,
                (0, 255, 255),
                2,
            )
            if save:
                if writer is None:
                    h, w = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot save MP4; check path and codec")
                writer.write(image)
            frames += 1
            if frames % 30 == 0:
                print(f"Processed {frames} frames", flush=True)
            if show_window:
                cv2.imshow("Sports ball - Q/ESC to stop", image)
                if cv2.waitKey(1) & 255 in (ord("q"), 27):
                    break
        if not frames:
            raise RuntimeError("No decoded video frames")
        summary = dict(
            frames=frames,
            observed_frames=observed_frames,
            tracked_frames=tracked_frames,
            tracked_boxes=tracked_boxes,
            observation_fraction=observed_frames / frames,
            tracked_fraction=tracked_frames / frames,
            tracks=[list(x) for x in sorted(identities)],
            processing_fps=frames / max(time.perf_counter() - started, 1e-9),
            python=platform.python_version(),
            model=str(model_path),
            model_sha256=sha256(model_path),
            input_sha256=sha256(video),
            ball_class=ball_class,
            image_size=image_size,
            confidence=confidence,
            device=device,
            effective_threads=torch.get_num_threads(),
            camera_compensation=camera_compensation,
            camera_motion_states=dict(states),
            trail_seconds=trail_seconds,
            trail_smoothing=trail_smoothing,
            notes="Coverage is not accuracy. sports ball does not distinguish basketball/football. IDs are scoped to a shot. Ultralytics track output may exclude unconfirmed detections.",
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if table_file is not None:
            table_file.close()
        if show_window:
            cv2.destroyAllWindows()
