"""Close-range basketball observations with full-frame context and local recovery."""

import cv2
import numpy as np
from .motion import CameraMotion


def iou(a, b):
    intersection = np.maximum(
        0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])
    ).prod()
    return float(
        intersection
        / max(
            1,
            np.maximum(0, a[2:] - a[:2]).prod()
            + np.maximum(0, b[2:] - b[:2]).prod()
            - intersection,
        )
    )


def suppress_equipment(candidates, equipment):
    # Reject net/rim-shaped duplicate ball boxes, not every ball near the basket.
    return [
        value
        for value in candidates
        if not any(iou(value[0], box) > 0.35 for box in equipment)
    ]


class CloseupDetector:
    def __init__(self, model, device, classes):
        self.model, self.device, self.classes = model, device, classes
        self.equipment_classes = [
            k for k, v in model.names.items() if v.lower() in ("rim", "net")
        ]
        self.center = None
        self.velocity = np.zeros(2)
        self.misses = 0
        self.diameter = None
        self.reference = None
        self.equipment = []
        self.equipment_memory = []
        self.camera = CameraMotion()

    def seed(self, image, observation):
        box = observation[0]
        self.center = (box[:2] + box[2:]) / 2
        self.velocity = np.zeros(2)
        self.diameter = float(max(box[2:] - box[:2]))
        self.misses = 0
        self.reference = self.appearance(image, box)

    @staticmethod
    def appearance(image, box):
        a, b, c, d = box.astype(int)
        patch = image[max(0, b) : max(0, d), max(0, a) : max(0, c)]
        if not patch.size:
            return None
        resized = cv2.resize(patch, (20, 20))
        return np.median(
            cv2.cvtColor(resized[5:15, 5:15], cv2.COLOR_BGR2HSV), axis=(0, 1)
        ).astype(float)

    def predict(self, image, region, global_search):
        x, y, r, b = region
        if r - x < 12 or b - y < 12:
            return [], []
        result = self.model.predict(
            image[y:b, x:r],
            classes=self.classes + self.equipment_classes,
            conf=0.06,
            imgsz=960 if global_search else 640,
            device=self.device,
            verbose=False,
            iou=0.4,
        )[0]
        balls = []
        equipment = []
        for row in result.boxes.data.cpu().numpy():
            box = row[:4].astype(float) + [x, y, x, y]
            if int(row[5]) in self.equipment_classes:
                if row[4] >= 0.08:
                    equipment.append(box)
            else:
                width, height = box[2:] - box[:2]
                maximum = max(32, min(image.shape[:2]) * 0.14)
                if self.diameter is not None:
                    maximum = min(maximum, self.diameter * 2.4)
                if (
                    min(width, height) >= 4
                    and max(width, height) <= maximum
                    and 0.5 < width / height < 1.9
                ):
                    balls.append(
                        (
                            box,
                            float(row[4]),
                            "closeup_full" if global_search else "closeup_roi",
                        )
                    )
        return balls, equipment

    def choose(self, image, candidates, prediction):
        scored = []
        for box, confidence, source in suppress_equipment(candidates, self.equipment):
            center = (box[:2] + box[2:]) / 2
            distance = 0 if prediction is None else np.linalg.norm(center - prediction)
            gate = max(60, (self.diameter or 10) * 3) + min(self.misses, 15) * 12
            if (
                prediction is not None
                and distance > gate
                and not (source == "closeup_full" and confidence >= 0.3)
            ):
                continue
            if confidence < (0.08 if source == "closeup_full" else 0.5):
                continue
            score = confidence - 0.2 * min(1, distance / max(gate, 1))
            color = self.appearance(image, box)
            if self.reference is not None and color is not None:
                hue = abs(color[0] - self.reference[0])
                hue = min(hue, 180 - hue)
                # Soft appearance evidence; lighting and spinning must not hard-reject a ball.
                score -= 0.12 * min(
                    1, hue / 35 + abs(color[1] - self.reference[1]) / 180
                )
            scored.append((score, box, confidence, source))
        return max(scored, key=lambda value: value[0]) if scored else None

    def observe(self, image):
        h, w = image.shape[:2]
        if self.misses >= 8:
            self.center = None
            self.velocity[:] = 0
            self.reference = None
            self.diameter = None
        prediction = (
            None
            if self.center is None
            else self.center + self.velocity * (self.misses + 1)
        )
        matrix, state = self.camera.update(image)
        remembered = []
        if matrix is not None:
            for box, age in self.equipment_memory:
                if age >= 15:
                    continue
                corners = np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
                moved = corners @ matrix[:,:2].T + matrix[:,2]
                remembered.append((np.r_[moved.min(axis=0),moved.max(axis=0)],age+1))
        candidates, equipment = self.predict(image, (0, 0, w, h), True)
        for box in equipment:
            remembered = [(old,age) for old,age in remembered if iou(box,old)<.6]
            remembered.append((box,0))
        self.equipment_memory = remembered[-40:]
        self.equipment = [box for box,age in self.equipment_memory]
        chosen = self.choose(image, candidates, prediction)
        # Global context prevents a tight crop from confusing clothing or net with a ball.
        # Recover low/missing full-frame detections at a second scale near the last observation.
        anchor = prediction
        if anchor is None and chosen is not None:
            anchor = (chosen[1][:2] + chosen[1][2:]) / 2
        if anchor is not None and (chosen is None or chosen[2] < 0.55):
            radius = max(110, (self.diameter or 20) * 4) + min(self.misses, 8) * 12
            region = (
                max(0, int(anchor[0] - radius)),
                max(0, int(anchor[1] - radius)),
                min(w, int(anchor[0] + radius)),
                min(h, int(anchor[1] + radius)),
            )
            local, extra_equipment = self.predict(image, region, False)
            self.equipment += extra_equipment
            chosen = self.choose(image, candidates + local, prediction)
        if chosen is None:
            self.misses += 1
            return None
        _, box, confidence, source = chosen
        center = (box[:2] + box[2:]) / 2
        if self.center is not None:
            displacement = (center - self.center) / max(1, self.misses + 1)
            self.velocity = 0.4 * self.velocity + 0.6 * displacement
        self.center = center
        self.diameter = (
            float(max(box[2:] - box[:2]))
            if self.diameter is None
            else 0.8 * self.diameter + 0.2 * float(max(box[2:] - box[:2]))
        )
        color = self.appearance(image, box)
        if color is not None:
            self.reference = (
                color if self.reference is None else 0.9 * self.reference + 0.1 * color
            )
        self.misses = 0
        return box, confidence, source
