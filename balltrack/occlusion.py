"""Bounded image-space forecasts. Never used as detector or ByteTrack observations."""

from collections import deque
import cv2
import numpy as np


class ShortGapPredictor:
    def __init__(self, max_seconds=0.3, display_seconds=0.7):
        if not 0 < max_seconds <= 0.5:
            raise ValueError("Prediction limit must be in (0, .5] seconds")
        self.max_seconds = max_seconds
        self.display_seconds = display_seconds
        self.history = deque(maxlen=6)
        self.path = deque()
        self.identity = None
        self.diameter = 20.0
        self.last_prediction = None

    def clear(self):
        self.history.clear()
        self.path.clear()
        self.identity = None
        self.last_prediction = None

    def forecast(self, frame, fps):
        if len(self.history) < 3:
            return None
        last_frame, last = self.history[-1]
        elapsed = frame - last_frame
        if elapsed < 1 or elapsed > max(1, round(self.max_seconds * fps)):
            return None
        times = np.array([f - last_frame for f, _ in self.history], float)
        positions = np.array([p for _, p in self.history], float)
        if np.any(np.diff(times) != 1):
            return None
        degree = 2 if len(times) >= 5 else 1
        coefficients = np.polyfit(times, positions, degree)
        fitted = np.stack(
            [np.polyval(coefficients[:, axis], times) for axis in range(2)], axis=1
        )
        error = float(np.sqrt(np.mean((fitted - positions) ** 2)))
        if error > max(3, self.diameter * 0.4):
            return None
        displacement = np.array(
            [
                np.polyval(coefficients[:, axis], elapsed)
                - np.polyval(coefficients[:, axis], 0)
                for axis in range(2)
            ]
        )
        point = last + displacement
        uncertainty = max(2, error) + 1.5 * elapsed + 0.15 * elapsed * elapsed
        if uncertainty > max(20, self.diameter * 1.2):
            return None
        return point, uncertainty, elapsed / fps

    def update(self, frame, fps, matrix, observation=None, shape=None):
        # Express all stored observed/predicted points in the current image coordinates.
        if matrix is None:
            self.clear()
        else:
            self.history = deque(
                ((f, matrix[:, :2] @ p + matrix[:, 2]) for f, p in self.history),
                maxlen=6,
            )
            self.path = deque(
                (f, matrix[:, :2] @ p + matrix[:, 2]) for f, p in self.path
            )
        while self.path and frame - self.path[0][0] > round(self.display_seconds * fps):
            self.path.popleft()
        predicted = self.forecast(frame, fps)
        if observation is not None:
            identity, center, diameter = observation
            center = np.array(center, float)
            if identity != self.identity:
                self.clear()
            elif self.history and frame != self.history[-1][0] + 1:
                if predicted is None or np.linalg.norm(center - predicted[0]) > max(
                    25, 3 * diameter
                ):
                    self.path.clear()
                elif self.path:
                    self.path.append((frame, center.copy()))
                self.history.clear()
            self.identity = identity
            self.diameter = float(diameter)
            self.history.append((frame, center.copy()))
            self.last_prediction = None
            return None
        if predicted is None:
            self.last_prediction = None
            return None
        point, uncertainty, horizon = predicted
        if shape is not None and not (
            0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]
        ):
            self.last_prediction = None
            return None
        if not self.path or self.path[-1][0] != frame - 1:
            self.path.clear()
            self.path.append((self.history[-1][0], self.history[-1][1].copy()))
        self.path.append((frame, point.copy()))
        self.last_prediction = point.copy()
        return dict(
            id=self.identity,
            center=point.copy(),
            uncertainty_px=uncertainty,
            horizon_s=horizon,
        )

    def draw(self, image, prediction):
        # Dashed orange = estimated segment; solid magenta remains observed trajectory.
        traveled = 0.0
        for (_, a), (_, b) in zip(self.path, list(self.path)[1:]):
            length = float(np.linalg.norm(b - a))
            offset = 0.0
            while offset < length - 1e-6:
                phase = (traveled + offset) % 12.0
                visible = phase < 7.0
                step = min(length - offset, (7.0 if visible else 12.0) - phase)
                if step < 1e-6:
                    offset += 1e-6
                    continue
                if visible:
                    p = a + (b - a) * (offset / length)
                    q = a + (b - a) * ((offset + step) / length)
                    cv2.line(
                        image,
                        tuple(np.rint(p).astype(int)),
                        tuple(np.rint(q).astype(int)),
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                offset += step
            traveled += length
        if prediction is not None:
            point = tuple(np.rint(prediction["center"]).astype(int))
            cv2.circle(image, point, 5, (0, 165, 255), 2)
            cv2.circle(
                image, point, round(prediction["uncertainty_px"]), (0, 110, 180), 1
            )
            cv2.putText(
                image,
                f"PREDICTED ID:{prediction['id']} +{prediction['horizon_s']:.2f}s",
                (max(0, point[0] - 70), max(90, point[1] - 12)),
                0,
                0.5,
                (0, 165, 255),
                2,
            )
