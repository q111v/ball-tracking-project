"""Re-render saved observations without running YOLO: python -m validation.prediction.replay"""

from pathlib import Path
import csv
import cv2
import numpy as np
from balltrack.motion import CameraMotion
from balltrack.display_trail import DisplayTrail
from balltrack.occlusion import ShortGapPredictor


def run():
    root = Path("validation/prediction")
    rows = {int(r["frame"]): r for r in csv.DictReader((root / "tracks.csv").open())}
    cap = cv2.VideoCapture("videos/basketball_10s.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    camera = CameraMotion()
    trail = DisplayTrail(0.7, 0.8)
    predictor = ShortGapPredictor(0.3)
    frame = predicted = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            matrix, state = camera.update(image)
            row = rows[frame]
            tid = int(row["id"])
            box = np.array([float(row[k]) for k in ["x1", "y1", "x2", "y2"]])
            center = (box[:2] + box[2:]) / 2
            measured = None
            if row["source"] != "missing":
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(image, tuple(center.astype(int)), 3, (0, 0, 255), -1)
                cv2.putText(
                    image,
                    f'Ball ID:{tid} {float(row["confidence"]):.2f}',
                    (max(0, x1 - 40), max(50, y1 - 10)),
                    0,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                if tid >= 0:
                    measured = (tid, center, max(box[2:] - box[:2]))
            trail.update(
                frame,
                fps,
                matrix,
                tid,
                center if measured else None,
                max(box[2:] - box[:2]),
            )
            trail.draw(image, frame, fps)
            if row["source"] != "missing" and tid < 0:
                predictor.clear()
                prediction = None
            else:
                prediction = predictor.update(
                    frame, fps, matrix, measured, image.shape[:2]
                )
            predictor.draw(image, prediction)
            predicted += int(prediction is not None)
            cv2.putText(
                image,
                "Basketball | saved observations replay",
                (15, 30),
                0,
                0.6,
                (0, 255, 255),
                2,
            )
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(root / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError("Cannot write replay")
            writer.write(image)
            if frame in (63, 107):
                cv2.imwrite(str(root / f"preview{frame}.jpg"), image)
            frame += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    assert frame == 300
    print(f"Replayed {frame} frames; predictions on {predicted} frames")


if __name__ == "__main__":
    run()
