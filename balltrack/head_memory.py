"""Track semantic head exclusions through short pose-detector misses.

Optical flow is used ONLY to exclude heads, never to fabricate ball observations.
Forward/backward consistency and photometric error reject unsupported flow.
"""

import cv2
import numpy as np


class HeadMemory:
    def __init__(self, max_age=20):
        self.previous = None
        self.tracks = []
        self.max_age = max_age

    def update(self, image, detected):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retained = []
        scale = max(image.shape[:2]) / 1280
        if (
            self.previous is not None
            and self.previous.shape == gray.shape
            and self.tracks
        ):
            offsets = np.array(
                [(x, y) for y in (-0.4, 0, 0.4) for x in (-0.4, 0, 0.4)], np.float32
            )
            points = (
                np.concatenate(
                    [
                        center[None, :] + offsets * radius
                        for center, radius, age in self.tracks
                    ]
                )
                .astype(np.float32)
                .reshape(-1, 1, 2)
            )
            moved, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous,
                gray,
                points,
                None,
                winSize=(31, 31),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
            )
            if moved is not None:
                restored, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.previous,
                    moved,
                    None,
                    winSize=(31, 31),
                    maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        25,
                        0.01,
                    ),
                )
                if restored is not None:
                    valid = (
                        (status.ravel() > 0)
                        & (back_status.ravel() > 0)
                        & (error.ravel() < 25)
                        & (
                            np.linalg.norm(restored[:, 0] - points[:, 0], axis=1)
                            < max(1.5, 2.5 * scale)
                        )
                    )
                    for i, (center, radius, age) in enumerate(self.tracks):
                        mask = valid[i * 9 : (i + 1) * 9]
                        if mask.sum() < 3 or age >= self.max_age:
                            continue
                        displacement = (moved[:, 0] - points[:, 0])[
                            i * 9 : (i + 1) * 9
                        ][mask]
                        delta = np.median(displacement, axis=0)
                        if np.linalg.norm(delta) > max(30, 80 * scale):
                            continue
                        if np.median(
                            np.linalg.norm(displacement - delta, axis=1)
                        ) > max(2, radius * 0.5):
                            continue
                        position = center + delta
                        if (
                            0 <= position[0] < image.shape[1]
                            and 0 <= position[1] < image.shape[0]
                        ):
                            retained.append((position, radius, age + 1))
        # Fresh semantic detections take precedence; flow only bridges actual misses.
        merged = [
            (np.asarray(center, float), float(radius), 0) for center, radius in detected
        ]
        for item in retained:
            if not any(
                np.linalg.norm(item[0] - x[0]) < max(item[1], x[1]) * 0.6
                for x in merged
            ):
                merged.append(item)
        self.tracks = sorted(merged, key=lambda x: (x[2] > 0, -x[1]))[:160]
        self.previous = gray
        return [(center, radius) for center, radius, _ in self.tracks]
