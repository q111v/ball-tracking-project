from types import SimpleNamespace
import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes
from balltrack.general import sports_ball_class, process_video


def test_class_lookup():
    assert sports_ball_class({0: "person", 32: "sports ball"}) == 32
    assert sports_ball_class(["person", "sports ball"]) == 1
    with pytest.raises(ValueError):
        sports_ball_class({0: "basketball"})


def test_invalid_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_video(tmp_path / "missing.mp4", tmp_path / "out")


def test_multiball_export_and_cut_reset(tmp_path, monkeypatch):
    import balltrack.general as module
    import csv

    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120)
    )
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    resets = []
    arrays = [
        [[10, 20, 30, 40, 1, 0.8, 32], [70, 20, 90, 40, 2, 0.7, 32]],
        [[12, 20, 32, 40, 1, 0.8, 32], [72, 20, 92, 40, 2, 0.7, 32]],
        [],
        [[16, 20, 36, 40, 1, 0.8, 32]],
    ]

    class Model:
        names = {32: "sports ball"}
        predictor = SimpleNamespace(
            trackers=[SimpleNamespace(reset=lambda: resets.append(1))]
        )

        def __init__(self, *args):
            self.index = 0

        def add_callback(self, *args):
            pass

        def track(self, image, **kwargs):
            assert kwargs["persist"] is True and kwargs["classes"] == [32]
            assert kwargs["tracker"].endswith("bytetrack.yaml")
            data = arrays[self.index]
            self.index += 1
            boxes = Boxes(
                torch.tensor(data, dtype=torch.float32).reshape(-1, 7), image.shape[:2]
            )
            return [SimpleNamespace(boxes=boxes)]

    class Camera:
        def __init__(self, *args):
            self.index = 0

        def update(self, image):
            state = ["initial", "ok", "cut", "ok"][self.index]
            self.index += 1
            return (None if state in ("initial", "cut") else np.eye(2, 3)), state

    monkeypatch.setattr(module, "YOLO", Model)
    monkeypatch.setattr(module, "CameraMotion", Camera)
    result = process_video(video, tmp_path / "out", show_window=False)
    assert result["frames"] == 4 and result["tracked_frames"] == 3
    assert result["tracked_boxes"] == 5
    assert result["tracks"] == [[0, 1], [0, 2], [1, 1]]
    assert len(resets) == 1
    rows = list(csv.DictReader((tmp_path / "out/tracks.csv").open()))
    assert len(rows) == 6 and rows[0]["x1"] == "10.0"
    assert rows[4]["state"] == "missing"
    cap = cv2.VideoCapture(str(tmp_path / "out/annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    cap.release()
