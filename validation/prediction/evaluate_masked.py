"""python -m validation.prediction.evaluate_masked (from project root)."""

import csv, json
from pathlib import Path
import cv2
import numpy as np
from balltrack.motion import CameraMotion
from balltrack.occlusion import ShortGapPredictor


def run(
    video="videos/basketball_10s.mp4", csv_path="validation/closeup/after/tracks.csv"
):
    rows = {int(r["frame"]): r for r in csv.DictReader(Path(csv_path).open())}
    windows = [(40, 46), (205, 211)]
    camera = CameraMotion()
    predictor = ShortGapPredictor(0.3)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    errors = []
    details = []
    frame = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            matrix, _ = camera.update(image)
            row = rows[frame]
            tid = int(row["id"])
            box = np.array([float(row[k]) for k in ["x1", "y1", "x2", "y2"]])
            center = (box[:2] + box[2:]) / 2
            masked = any(a <= frame < b for a, b in windows)
            observation = (
                (tid, center, max(box[2:] - box[:2]))
                if tid >= 0 and not masked
                else None
            )
            if row["source"] != "missing" and tid < 0 and not masked:
                predictor.clear()
                prediction = None
            else:
                prediction = predictor.update(
                    frame, fps, matrix, observation, image.shape[:2]
                )
            if masked:
                assert tid >= 0 and row["source"] != "missing"
                error = None
                if prediction is not None:
                    error = float(np.linalg.norm(prediction["center"] - center))
                    errors.append(error)
                details.append(dict(frame=frame, error_px=error))
            frame += 1
    finally:
        cap.release()
    return dict(
        masked_windows=windows,
        masked_frames=len(details),
        predicted_frames=len(errors),
        rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        max_error_px=max(errors) if errors else None,
        details=details,
        notes="Artificially withheld previously detected centers, not independent manual ground truth or real occlusion accuracy. Two clear flight windows were chosen; bounce/contact and long gaps are not covered.",
    )


if __name__ == "__main__":
    result = run()
    Path("validation/prediction/masked_checks.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
