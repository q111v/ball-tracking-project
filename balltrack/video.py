"""Basketball file processing with bounded lossless cache and optional backward pass."""

import csv
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from .basketball import BasketballTracker, assign_observation, new_tracker
from .motion import CameraMotion
from .display_trail import DisplayTrail
from collections import Counter
from .temporal import select_trajectory, select_anchored_trajectory
from .runtime import configure_model
from .occlusion import ShortGapPredictor
from .scene import detect_scene

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def cache_write(path, image):
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("Could not cache a video frame")
    Path(path).write_bytes(data.tobytes())


def cache_read(path):
    image = cv2.imdecode(
        np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise RuntimeError("Could not decode cached frame")
    return image


def chunk_capacity(shape, requested=120, budget_mb=384):
    return max(1, min(requested, int(budget_mb * 1024 * 1024 / max(1, np.prod(shape)))))


def process_video(
    video,
    output,
    *,
    device="cpu",
    threads=2,
    trail=90,
    max_frames=0,
    show_window=True,
    save=True,
    profile="broadcast",
    mode="bidirectional",
    select_ball=False,
    initial_box=None,
    basketball_model=None,
    camera_compensation=True,
    trail_seconds=0.7,
    trail_smoothing=0.8,
    predict_occlusion=False,
    prediction_seconds=0.3,
):
    if mode not in ("auto", "forward", "bidirectional"):
        raise ValueError("PROCESSING_MODE must be forward or bidirectional")
    if profile not in ("auto", "broadcast", "closeup"):
        raise ValueError("SCENE must be broadcast or closeup")
    if video is None or not Path(video).is_file():
        raise ValueError("Basketball mode requires an existing video file")
    if not 0 < trail_seconds <= 10 or not 0 < trail_smoothing <= 1:
        raise ValueError("Invalid trail duration or smoothing")
    if threads < 1 or max_frames < 0 or trail < 1:
        raise ValueError("Invalid thread/frame/trail arguments")
    if select_ball and (not show_window):
        raise ValueError("Selecting a ball requires a display window")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    requested_profile = profile
    pose_model = configure_model(YOLO(str(ROOT / "models/yolov8n-pose.pt")), threads)
    scene_details = []
    if profile == "auto":
        profile, scene_details = detect_scene(video, pose_model, device)
        print(f"Selected scene: {profile}", flush=True)
    if mode == "auto":
        mode = "forward" if profile == "closeup" else "bidirectional"
    default_model = "basketball_closeup.pt" if profile == "closeup" else "basketball.pt"
    model = Path(basketball_model or ROOT / "models" / default_model)
    if not model.is_file():
        raise FileNotFoundError(
            f"Missing model: {model}. Download the complete project package."
        )
    detector = BasketballTracker(
        model,
        pose=pose_model,
        tracking=False,
        device=device,
        profile=profile,
    )
    configure_model(detector.model, threads)
    configure_model(detector.pose, threads)
    cap = cv2.VideoCapture(str(video))
    writer = None
    log = None
    trace = None
    frames = observed = tracked = 0
    predicted_frames = 0
    prediction_log = None
    predictor = (
        ShortGapPredictor(prediction_seconds, trail_seconds)
        if predict_occlusion
        else None
    )
    ids = set()
    last_id = None
    display_trail = DisplayTrail(trail_seconds, trail_smoothing)
    camera = CameraMotion(camera_compensation)
    motion_states = Counter()
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 30.0
        identity_tracker = new_tracker(fps)
        log = (output / "tracks.csv").open("w", newline="", encoding="utf-8")
        table = csv.writer(log)
        table.writerow(
            [
                "frame",
                "time_s",
                "id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "source",
                "pass",
                "processing_s",
            ]
        )
        trace = (output / "observations.jsonl").open("w", encoding="utf-8")
        if predictor is not None:
            prediction_log = (output / "predictions.csv").open(
                "w", newline="", encoding="utf-8"
            )
            prediction_table = csv.writer(prediction_log)
            prediction_table.writerow(
                [
                    "frame",
                    "time_s",
                    "id",
                    "center_x",
                    "center_y",
                    "uncertainty_px",
                    "horizon_s",
                    "source",
                ]
            )
        print(
            f"Basketball processing started: {mode}. "
            + (
                "Please wait for each offline chunk."
                if mode == "bidirectional"
                else "Processing frames in order."
            ),
            flush=True,
        )
        manual_initialization = bool(select_ball or initial_box)
        chosen_initial_box = None
        started = time.perf_counter()
        eof = False
        read_frames = 0
        while not eof and (not max_frames or read_frames < max_frames):
            with tempfile.TemporaryDirectory(
                prefix=".ball_frames_", dir=output
            ) as temporary:
                paths = []
                forward = []
                durations = []
                head_cache = []
                limit = 1 if mode == "forward" else 120
                for _ in range(limit):
                    if max_frames and read_frames >= max_frames:
                        break
                    ok, image = cap.read()
                    if not ok:
                        eof = True
                        break
                    if not paths:
                        limit = 1 if mode == "forward" else chunk_capacity(image.shape)
                    if read_frames == 0 and (select_ball or initial_box):
                        if initial_box:
                            x, y, w, h = initial_box
                        else:
                            print(
                                "Draw a rectangle around the basketball, then press ENTER. ESC cancels.",
                                flush=True,
                            )
                            x, y, w, h = cv2.selectROI(
                                "Select basketball, ENTER confirms, ESC cancels",
                                image.copy(),
                                False,
                                False,
                            )
                            cv2.destroyWindow(
                                "Select basketball, ENTER confirms, ESC cancels"
                            )
                        if (
                            w <= 4
                            or h <= 4
                            or x < 0
                            or y < 0
                            or x + w > image.shape[1]
                            or y + h > image.shape[0]
                        ):
                            raise ValueError(
                                "Invalid or cancelled initial ball selection"
                            )
                        chosen_initial_box = list(map(float, [x, y, w, h]))
                        detector.seed(
                            image,
                            (
                                np.array([x, y, x + w, y + h], float),
                                1.0,
                                "manual_initialization",
                            ),
                        )
                    path = Path(temporary) / f"{len(paths):04d}.png"
                    cache_write(path, image)
                    paths.append(path)
                    tick = time.perf_counter()
                    forward.append(detector.observe(image))
                    head_cache.append(list(detector.heads))
                    durations.append(time.perf_counter() - tick)
                    read_frames += 1
                    if read_frames % 20 == 0:
                        print(f"Forward detection: frame {read_frames}", flush=True)
                    if len(paths) >= limit:
                        break
                if not paths:
                    break
                backward = [None] * len(paths)
                if mode == "bidirectional":
                    last = next(
                        (
                            i
                            for i in range(len(paths) - 1, -1, -1)
                            if forward[i] is not None
                        ),
                        None,
                    )
                    if last is not None:
                        reverse = BasketballTracker(
                            detector.model,
                            pose=detector.pose,
                            tracking=False,
                            device=device,
                            profile=profile,
                        )
                        reverse.seed(cache_read(paths[last]), forward[last])
                        for i in range(last, -1, -1):
                            image = cache_read(paths[i])
                            tick = time.perf_counter()
                            backward[i] = reverse.observe(image, heads=head_cache[i])
                            durations[i] += time.perf_counter() - tick
                            if i % 20 == 0:
                                print(
                                    f"Backward detection: {i} frames left in chunk",
                                    flush=True,
                                )
                    shape = cache_read(paths[0]).shape
                    selected = (
                        select_anchored_trajectory
                        if manual_initialization
                        else select_trajectory
                    )(forward, backward, max(shape[:2]) / 1280)
                else:
                    selected = forward

                def encode_observation(value):
                    return (
                        None
                        if value is None
                        else dict(
                            bbox=list(map(float, value[0])),
                            confidence=float(value[1]),
                            source=value[2],
                        )
                    )

                for i, observation in enumerate(selected):
                    trace.write(
                        json.dumps(
                            dict(
                                frame=frames,
                                forward=encode_observation(forward[i]),
                                backward=encode_observation(backward[i]),
                                selected=encode_observation(observation),
                            )
                        )
                        + "\n"
                    )
                    image = cache_read(paths[i])
                    tick = time.perf_counter()
                    matrix, motion_state = camera.update(
                        image,
                        head_cache[i],
                        None if observation is None else observation[0],
                    )
                    motion_states[motion_state] += 1
                    # Drop association state on a detected shot change.
                    if motion_state == "cut":
                        identity_tracker = new_tracker(fps)
                        last_id = None
                    tid = assign_observation(
                        identity_tracker, observation, image.shape[:2]
                    )
                    display_center = None
                    diameter = 10
                    if observation is not None:
                        box, confidence, source = observation
                        observed += 1
                        center = tuple(map(int, (box[:2] + box[2:]) / 2))
                        if tid >= 0:
                            ids.add(tid)
                            last_id = tid
                            tracked += 1
                            display_center = (box[:2] + box[2:]) / 2
                            diameter = max(box[2] - box[0], box[3] - box[1])
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(image, center, 3, (0, 0, 255), -1)
                        cv2.putText(
                            image,
                            f'Ball ID:{tid if tid>=0 else "?"} {confidence:.2f}',
                            (max(0, x1 - 40), max(50, y1 - 10)),
                            0,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        origin = "forward" if forward[i] is observation else "backward"
                    else:
                        box = [0] * 4
                        confidence = 0.0
                        source = "missing"
                        origin = "none"
                        cv2.putText(
                            image,
                            f"ID:{last_id} NOT OBSERVED",
                            (15, 65),
                            0,
                            0.7,
                            (0, 180, 255),
                            2,
                        )
                    display_trail.update(
                        frames, fps, matrix, tid, display_center, diameter
                    )
                    display_trail.draw(image, frames, fps)
                    if predictor is not None:
                        if observation is not None and tid < 0:
                            predictor.clear()
                            prediction = None
                        else:
                            measured = (
                                None
                                if observation is None
                                else (tid, display_center, diameter)
                            )
                            prediction = predictor.update(
                                frames,
                                fps,
                                matrix,
                                measured,
                                image.shape[:2],
                                blocked_heads=head_cache[i],
                            )
                        predictor.draw(image, prediction)
                        if prediction is not None:
                            predicted_frames += 1
                            prediction_table.writerow(
                                [
                                    frames,
                                    frames / fps,
                                    prediction["id"],
                                    *map(float, prediction["center"]),
                                    prediction["uncertainty_px"],
                                    prediction["horizon_s"],
                                    "predicted",
                                ]
                            )
                    rate = (frames + 1) / max(time.perf_counter() - started, 1e-9)
                    cv2.putText(
                        image,
                        f"Basketball | {mode} | process FPS:{rate:.1f}",
                        (15, 30),
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
                                raise RuntimeError(
                                    "Cannot write MP4; check output path and codec"
                                )
                        writer.write(image)
                    duration = durations[i] + time.perf_counter() - tick
                    table.writerow(
                        [
                            frames,
                            frames / fps,
                            tid,
                            *map(float, box),
                            confidence,
                            source,
                            origin,
                            duration,
                        ]
                    )
                    frames += 1
                    if not (not show_window):
                        cv2.imshow("Basketball - Q/ESC to quit", image)
                        if cv2.waitKey(1) & 255 in (ord("q"), 27):
                            eof = True
                            break
                print(f"Processed {frames} frames, {observed} observations", flush=True)
        if not frames:
            raise RuntimeError("No decoded frames")
        elapsed = time.perf_counter() - started
        summary = dict(
            frames=frames,
            observed_frames=observed,
            yolo_observed_frames=observed,
            appearance_recovered_frames=0,
            recovery_notes="All observed boxes come from YOLO. Experimental color recovery was rejected and is not used.",
            predicted_frames=predicted_frames,
            predict_occlusion=predict_occlusion,
            prediction_seconds=prediction_seconds,
            prediction_notes="Predictions are short image-space extrapolations, not detections or calibrated confidence intervals. They never update ByteTrack or observation counts.",
            tracked_frames=tracked,
            observation_fraction=observed / frames,
            tracked_fraction=tracked / frames,
            unique_ids=sorted(ids),
            processing_fps=frames / elapsed,
            python=platform.python_version(),
            initialization="manual" if manual_initialization else "automatic",
            initial_box=chosen_initial_box,
            device=device,
            threads=threads,
            effective_threads=torch.get_num_threads(),
            mode=mode,
            profile=profile,
            requested_profile=requested_profile,
            scene_selection=scene_details,
            camera_compensation=camera_compensation,
            camera_motion_states=dict(motion_states),
            trail_seconds=trail_seconds,
            trail_smoothing=trail_smoothing,
            trail_resets=display_trail.resets,
            model=str(model),
            detector_sha256=sha256_file(model),
            input_sha256=sha256_file(video),
            notes="Coverage is not accuracy. Missing frames have no invented boxes. Bidirectional mode is offline and may delay display by one chunk.",
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if log is not None:
            log.close()
        if trace is not None:
            trace.close()
        if prediction_log is not None:
            prediction_log.close()
        if not (not show_window):
            cv2.destroyAllWindows()
