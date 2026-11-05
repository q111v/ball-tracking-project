from collections import deque


class Trails:
    """Bounded observed histories; never invent detections during occlusion."""
    def __init__(self, length=90, max_gap=30):
        self.length, self.max_gap = length, max_gap
        self.points, self.last = {}, {}

    def update(self, frame, observations):
        for tid in list(self.last):
            if frame - self.last[tid] > self.max_gap:
                self.last.pop(tid)
                self.points.pop(tid)
        for tid, center in observations:
            self.points.setdefault(tid, deque(maxlen=self.length)).append((frame, center))
            self.last[tid] = frame

    def segments(self, tid):
        points = list(self.points.get(tid, []))
        return [(a[1], b[1]) for a, b in zip(points, points[1:]) if b[0] == a[0] + 1]
