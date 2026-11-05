"""Rebuild the small same-video dataset and fine-tune the public weights.

Run from any directory: python tools/train_closeup.py --epochs 18
Validation frames are adjacent frames from the same video, not independent footage.
"""

import argparse
import json
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "training_results"
    data = output / "dataset"
    data.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "training/closeup_annotations.json").read_text())
    cap = cv2.VideoCapture(str(root / "videos/basketball_10s.mp4"))
    try:
        for entry in manifest:
            split, frame = entry["split"], entry["frame"]
            for kind in ("images", "labels"):
                (data / kind / split).mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read source frame {frame}")
            name = f"frame{frame:03d}"
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            (data / "images" / split / (name + ".jpg")).write_bytes(encoded.tobytes())
            label = root / "training/closeup_labels" / split / (name + ".txt")
            (data / "labels" / split / (name + ".txt")).write_bytes(label.read_bytes())
    finally:
        cap.release()
    torch.set_num_threads(2)
    model = YOLO(str(root / "models/basketball_base.pt"))
    names = "".join(f'  {i}: "{name}"\n' for i, name in model.names.items())
    config = data / "data.yaml"
    config.write_text(
        f"path: {data.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + names
    )
    model.train(
        data=str(config),
        epochs=args.epochs,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        freeze=10,
        amp=False,
        seed=17,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.2,
        degrees=3,
        translate=0.08,
        scale=0.25,
        fliplr=0.3,
        plots=False,
        save=True,
        patience=18,
        project=str(output),
        name="held_ball",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
