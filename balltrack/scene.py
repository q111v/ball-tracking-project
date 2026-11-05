"""Choose a scene from visible person scale/count, never from filenames."""

import cv2
import numpy as np


def scene_from_people(boxes, height):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    sizes = (boxes[:, 3] - boxes[:, 1]) / height
    sizes = sizes[sizes >= 0.12]
    if len(sizes) and len(sizes) <= 3 and max(sizes) >= 0.32:
        return "closeup"
    return "broadcast"


def detect_scene(video, pose, device):
    cap = cv2.VideoCapture(str(video))
    votes = []
    details = []
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video for scene selection")
        for frame in (0, 10, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                continue
            result = pose.predict(
                image, conf=0.25, imgsz=960, device=device, verbose=False
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            vote = scene_from_people(boxes, image.shape[0])
            votes.append(vote)
            details.append(dict(frame=frame, scene=vote, people=len(boxes)))
        if not votes:
            raise RuntimeError("No frames for scene selection")
        return (
            "closeup" if votes.count("closeup") > len(votes) / 2 else "broadcast"
        ), details
    finally:
        cap.release()
