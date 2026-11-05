"""Basketball observations from public YOLO weights; no clip-specific coordinates.

The detector supports bounded-memory forward/backward video processing. All
exported boxes have image detection evidence, not interpolated trajectories.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from .head_memory import HeadMemory
from .closeup import CloseupDetector


def head_regions(keypoints, person):
    points = np.asarray(keypoints)
    face = points[:5]
    valid = face[face[:, 2] > 0.1]
    height = person[3] - person[1]
    if len(valid):
        center = np.average(valid[:, :2], axis=0, weights=valid[:, 2]) - np.array(
            [0, height * 0.04]
        )
    elif len(points) > 6 and np.all(points[5:7, 2] > 0.3):
        center = points[5:7, :2].mean(axis=0) - np.array([0, height * 0.13])
    else:
        return []
    return [(center, max(12, height * 0.14))]


def association_box(box, scale=1.0):
    box = np.asarray(box, dtype=float)
    center = (box[:2] + box[2:]) / 2
    half = max(128 * scale, float(np.max(box[2:] - box[:2])) * 2)
    return np.r_[center - half, center + half]


def new_tracker(fps=30):
    return BYTETracker(
        SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.35,
            track_buffer=45,
            match_thresh=0.99,
            fuse_score=False,
        ),
        frame_rate=fps,
    )


def assign_observation(tracker, observation, shape):
    rows = []
    if observation is not None:
        box, confidence, _ = observation
        rows = [[*association_box(box, max(shape) / 1280), confidence, 0]]
    tracks = tracker.update(Boxes(np.asarray(rows, np.float32).reshape(-1, 6), shape))
    return int(tracks[0, 4]) if len(tracks) else -1


def fuse_observations(forward, backward):
    """Choose measured evidence only. Disagreement is resolved by confidence."""
    if forward is None:
        return backward
    if backward is None:
        return forward
    return forward if forward[1] >= backward[1] else backward


class BasketballTracker:
    def __init__(
        self,
        model,
        device="cpu",
        roi_size=256,
        pose=None,
        tracking=True,
        profile="auto",
    ):
        self.model = model if isinstance(model, YOLO) else YOLO(str(model))
        self.pose = (
            pose
            if pose is not None
            else YOLO(str(Path(model).parent / "yolov8n-pose.pt"))
        )
        self.classes = [
            k
            for k, v in self.model.names.items()
            if v.lower() in ("ball", "basketball", "sports ball")
        ]
        if not self.classes:
            raise ValueError("Model has no ball class")
        if profile not in ("auto", "broadcast", "closeup"):
            raise ValueError("Unknown basketball profile")
        self.manual_lock = False
        self.head_memory = HeadMemory()
        self.profile = profile
        self.device = device
        self.roi_size = roi_size
        self.reset()
        self.tracker = new_tracker() if tracking else None
        self.closeup = (
            CloseupDetector(self.model, device, self.classes)
            if profile == "closeup"
            else None
        )

    def reset(self):
        self.center = None
        self.velocity = np.zeros(2)
        self.misses = 0
        self.template = None
        self.diameter = None
        self.color = None
        self.heads = []

    def seed(self, image, observation):
        if getattr(self, "closeup", None) is not None:
            self.closeup.seed(image, observation)
        box, _, source = observation
        if source == "manual_initialization":
            self.manual_lock = True
        self.center = (box[:2] + box[2:]) / 2
        self.velocity = np.zeros(2)
        self.misses = 0
        self.diameter = float(max(box[2:] - box[:2]))
        a, b, c, d = box.astype(int)
        patch = image[max(0, b) : d, max(0, a) : c]
        self.template = cv2.resize(patch, (24, 24)) if patch.size else None
        if self.template is not None:
            self.color = np.median(
                cv2.cvtColor(self.template[8:16, 8:16], cv2.COLOR_BGR2HSV), axis=(0, 1)
            )

    def candidates(self, image, region, global_search=False):
        x1, y1, x2, y2 = region
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        out = self.model.predict(
            image[y1:y2, x1:x2],
            classes=self.classes,
            conf=0.08,
            imgsz=1280 if global_search else 640,
            device=self.device,
            verbose=False,
            iou=0.4,
        )[0]
        candidates = []
        scale = max(image.shape[:2]) / 1280
        maximum = (100 if self.profile == "broadcast" else 500) * scale
        if self.diameter is not None:
            maximum = min(maximum, self.diameter * 2.5)
        for b in out.boxes.data.cpu().numpy():
            box = b[:4] + np.array([x1, y1, x1, y1])
            w, h = box[2:] - box[:2]
            if min(w, h) < 4 * scale or max(w, h) > maximum or not 0.45 < w / h < 2.2:
                continue
            candidates.append(
                (box, float(b[4]), "yolo_global" if global_search else "yolo_roi")
            )
        return candidates

    def score_candidates(self, image, candidates, prediction):
        scored = []
        scale = max(image.shape[:2]) / 1280
        for box, conf, source in candidates:
            center = (box[:2] + box[2:]) / 2
            diameter = max(box[2:] - box[:2])
            # Broadcast initialization uses the central court band, excluding most
            # scoreboard/audience graphics. A manually selected seed overrides this.
            if (
                prediction is None
                and self.profile == "broadcast"
                and not 0.18 * image.shape[0] <= center[1] <= 0.80 * image.shape[0]
            ):
                continue
            if any(
                diameter < radius * 4 and np.linalg.norm(center - head) < radius
                for head, radius in self.heads
            ):
                continue
            distance = 0 if prediction is None else np.linalg.norm(center - prediction)
            extra = (
                self.misses * 10 * scale
                if self.manual_lock
                else min(self.misses, 8) * 5 * scale
            )
            gate = max(80 * scale, (self.diameter or 0) * 3) + extra
            if prediction is not None and distance > gate:
                continue
            score = conf - distance / max(180 * scale, (self.diameter or 1) * 6)
            if self.template is not None:
                a, b, c, d = box.astype(int)
                patch = image[
                    max(0, b) : min(image.shape[0], d),
                    max(0, a) : min(image.shape[1], c),
                ]
                if patch.size:
                    resized = cv2.resize(patch, (24, 24))
                    color = np.median(
                        cv2.cvtColor(resized[8:16, 8:16], cv2.COLOR_BGR2HSV),
                        axis=(0, 1),
                    )
                    hue = abs(color[0] - self.color[0])
                    # Reject orange rim fittings/floor patches with incompatible appearance.
                    # Reference is learned from the selected ball, never a fixed brown range.
                    if self.profile != "closeup" and (
                        min(hue, 180 - hue) > 18
                        or abs(color[1] - self.color[1]) > 105
                        or color[2] > min(255, self.color[2] + 70)
                        or color[2] < self.color[2] * 0.3
                    ):
                        continue
                    similarity = float(
                        cv2.matchTemplate(resized, self.template, cv2.TM_CCOEFF_NORMED)[
                            0, 0
                        ]
                    )
                    score += 0.15 * similarity
            scored.append((score, box, conf, source))
        return scored

    def observe(self, image, heads=None):
        if self.closeup is not None:
            self.heads = []
            return self.closeup.observe(image)
        h, w = image.shape[:2]
        scale = max(h, w) / 1280
        if self.misses >= 30 and not self.manual_lock:
            self.reset()
        if heads is None:
            pose = self.pose.predict(
                image, conf=0.10, imgsz=1280, device=self.device, verbose=False
            )[0]
            self.heads = []
            if pose.keypoints is not None:
                for keypoints, person in zip(
                    pose.keypoints.data.cpu().numpy(), pose.boxes.xyxy.cpu().numpy()
                ):
                    self.heads.extend(head_regions(keypoints, person))
            self.heads = self.head_memory.update(image, self.heads)
        else:
            self.heads = heads
        prediction = None if self.center is None else self.center + self.velocity
        global_search = self.center is None
        if global_search:
            global_candidates = self.score_candidates(
                image, self.candidates(image, (0, 0, w, h), True), None
            )
            scored = []
            # Verify the best global proposals at a second scale before choosing an identity.
            for _, anchor, global_conf, _ in sorted(
                global_candidates, key=lambda x: x[0], reverse=True
            )[:3]:
                center = (anchor[:2] + anchor[2:]) / 2
                radius = max(
                    self.roi_size * scale / 2, float(max(anchor[2:] - anchor[:2])) * 2
                )
                region = (
                    max(0, int(center[0] - radius)),
                    max(0, int(center[1] - radius)),
                    min(w, int(center[0] + radius)),
                    min(h, int(center[1] + radius)),
                )
                for _, box, confidence, source in self.score_candidates(
                    image, self.candidates(image, region), None
                ):
                    if np.linalg.norm((box[:2] + box[2:]) / 2 - center) > max(
                        15 * scale, max(anchor[2:] - anchor[:2]) * 0.6
                    ):
                        continue
                    verified = float(np.sqrt(global_conf * confidence))
                    scored.append((verified, box, verified, "yolo_global_verified"))
        else:
            scored = []
            # The crop grows with the observed ball and image resolution, not fixed clip positions.
            radius = (
                max(self.roi_size * scale / 2, (self.diameter or 0) * 2)
                + min(self.misses, 5) * 12 * scale
            )
            for factor in (1.0, 0.75, 1.5):
                region = (
                    max(0, int(prediction[0] - radius * factor)),
                    max(0, int(prediction[1] - radius * factor)),
                    min(w, int(prediction[0] + radius * factor)),
                    min(h, int(prediction[1] + radius * factor)),
                )
                scored.extend(
                    self.score_candidates(
                        image, self.candidates(image, region), prediction
                    )
                )
                if scored and max(x[2] for x in scored) >= 0.6:
                    break
            if self.misses >= 8 and not scored:
                scored = self.score_candidates(
                    image, self.candidates(image, (0, 0, w, h), True), prediction
                )
        if not scored:
            self.misses += 1
            return None
        _, box, confidence, source = max(scored, key=lambda x: x[0])
        if self.center is not None:
            displacement = (box[:2] + box[2:]) / 2 - self.center
            self.velocity = 0.5 * self.velocity + 0.5 * displacement / max(
                1, self.misses + 1
            )
        observation = (box, confidence, source)
        if self.template is None:
            self.seed(image, observation)
        self.center = (box[:2] + box[2:]) / 2
        self.diameter = (
            float(max(box[2:] - box[:2]))
            if self.diameter is None
            else 0.9 * self.diameter + 0.1 * float(max(box[2:] - box[:2]))
        )
        self.misses = 0
        return observation

    def step(self, image):
        observation = self.observe(image)
        if self.tracker is None:
            raise RuntimeError("Tracking disabled on this observer")
        return observation, assign_observation(
            self.tracker, observation, image.shape[:2]
        )
