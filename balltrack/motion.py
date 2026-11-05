"""Estimate background motion; all matrices map previous pixels to current pixels."""

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, enabled=True, width=640):
        self.enabled = enabled
        self.width = width
        self.previous = None
        self.mask = None

    def update(self, image, heads=(), ball=None):
        height, width = image.shape[:2]
        scale = min(1.0, self.width / width)
        gray = cv2.cvtColor(
            cv2.resize(image, (round(width * scale), round(height * scale))),
            cv2.COLOR_BGR2GRAY,
        )
        mask = np.full(gray.shape, 255, np.uint8)
        h, w = gray.shape
        # Ignore broadcast graphics at the top and bottom edges.
        mask[: int(h * 0.15)] = 0
        mask[int(h * 0.85) :] = 0
        for center, radius in heads:
            x, y = np.asarray(center) * scale
            r = radius * scale
            cv2.rectangle(
                mask,
                (int(x - 2 * r), int(y - r)),
                (int(x + 2 * r), int(y + 7 * r)),
                0,
                -1,
            )
        if ball is not None:
            x1, y1, x2, y2 = np.asarray(ball) * scale
            cv2.rectangle(
                mask, (int(x1 - 10), int(y1 - 10)), (int(x2 + 10), int(y2 + 10)), 0, -1
            )
        previous, old_mask = self.previous, self.mask
        self.previous, self.mask = gray, mask
        if previous is None:
            return None, "initial"
        if previous.shape != gray.shape:
            return None, "reset"
        # Large appearance changes indicate a cut, including when compensation is disabled.
        hist1 = cv2.calcHist([previous], [0], None, [32], [0, 256])
        hist2 = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        if cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) > 0.55:
            return None, "cut"
        if not self.enabled:
            return np.eye(2, 3), "disabled"
        points = cv2.goodFeaturesToTrack(previous, 350, 0.01, 10, mask=old_mask)
        if points is None or len(points) < 12:
            return None, "unreliable"
        current, ok, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(21, 21), maxLevel=3
        )
        if current is None:
            return None, "unreliable"
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous, current, None, winSize=(21, 21), maxLevel=3
        )
        if back is None:
            return None, "unreliable"
        good = (
            (ok.ravel() == 1)
            & (back_ok.ravel() == 1)
            & (np.linalg.norm(points - back, axis=2).ravel() < 1.5)
        )
        xy = current.reshape(-1, 2)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        indices = np.flatnonzero(good & inside)
        indices = indices[
            mask[xy[indices, 1].astype(int), xy[indices, 0].astype(int)] > 0
        ]
        if len(indices) < 12:
            return None, "unreliable"
        before = points.reshape(-1, 2)[indices]
        after = xy[indices]
        matrix, inliers = cv2.estimateAffinePartial2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=1500
        )
        if matrix is None or inliers.sum() < 10 or inliers.mean() < 0.5:
            return None, "unreliable"
        kept = before[inliers.ravel().astype(bool)]
        # Reject a transform supported only by one small moving player.
        if np.ptp(kept[:, 0]) < w * 0.2 or np.ptp(kept[:, 1]) < h * 0.15:
            return None, "unreliable"
        zoom = np.linalg.norm(matrix[0, :2])
        angle = abs(np.arctan2(matrix[1, 0], matrix[0, 0]))
        if (
            not 0.9 < zoom < 1.1
            or angle > 0.12
            or np.linalg.norm(matrix[:, 2]) > w * 0.2
        ):
            return None, "unreliable"
        matrix[:, 2] /= scale
        return matrix, "ok"
