"""Short fading trails in current-frame coordinates, independent of CSV measurements."""

from collections import deque
import cv2
import numpy as np


class DisplayTrail:
    def __init__(self, seconds=0.7, smooth=0.8):
        self.seconds = seconds
        self.smooth = smooth
        self.points = deque()
        self.resets = 0

    def update(self, frame, fps, matrix, identity, center, diameter=10):
        if matrix is None:
            if self.points:
                self.resets += 1
            self.points.clear()
        else:
            self.points = deque(
                (f, tid, matrix[:, :2] @ p + matrix[:, 2]) for f, tid, p in self.points
            )
        lifetime = max(1, round(self.seconds * fps))
        while self.points and frame - self.points[0][0] > lifetime:
            self.points.popleft()
        if center is None or identity < 0:
            return
        point = np.asarray(center, dtype=float)
        if self.points:
            last_frame, last_id, previous = self.points[-1]
            if (
                last_id != identity
                or frame - last_frame != 1
                or np.linalg.norm(point - previous) > max(60, diameter * 6)
            ):
                self.points.clear()
                self.resets += 1
            else:
                point = self.smooth * point + (1 - self.smooth) * previous
        self.points.append((frame, identity, point))

    def draw(self, image, frame, fps):
        lifetime = max(1, round(self.seconds * fps))
        h, w = image.shape[:2]
        for a, b in zip(self.points, list(self.points)[1:]):
            if b[0] != a[0] + 1 or a[1] != b[1]:
                continue
            p, q = np.rint(a[2]).astype(int), np.rint(b[2]).astype(int)
            # Clip before forming a small overlay; avoid copying a whole frame per segment.
            visible, p, q = cv2.clipLine((0, 0, w, h), tuple(p), tuple(q))
            if not visible:
                continue
            x1, y1 = max(0, min(p[0], q[0]) - 3), max(0, min(p[1], q[1]) - 3)
            x2, y2 = min(w, max(p[0], q[0]) + 4), min(h, max(p[1], q[1]) + 4)
            roi = image[y1:y2, x1:x2]
            overlay = roi.copy()
            cv2.line(
                overlay,
                (p[0] - x1, p[1] - y1),
                (q[0] - x1, q[1] - y1),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            alpha = max(0.08, 1 - (frame - a[0]) / lifetime)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
