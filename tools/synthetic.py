"""Deterministic tracker-only benchmark. Does not claim YOLO detects colored circles."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from balltrack.app import ROOT
from balltrack.evaluate import evaluate


def generate(output, buffer=30):
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT/'configs/bytetrack.yaml').read_text())
    cfg['track_buffer'] = buffer
    tracker = BYTETracker(SimpleNamespace(**cfg), frame_rate=30)
    video = cv2.VideoWriter(str(output/'synthetic.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,360))
    if not video.isOpened():
        raise RuntimeError('Cannot write synthetic video')
    truth, predictions = [], {}
    try:
        for f in range(120):
            image = np.full((360,640,3),35,dtype=np.uint8)
            objects, detections = [], []
            phase = 'before' if f < 45 else 'occluded' if f < 53 else 'after'
            for tid, y in ((1,100),(2,260)):
                x = 60+3*f
                box = [x-15,y-15,x+15,y+15]
                visible = not (tid == 1 and 45 <= f < 53)
                objects.append(dict(id=tid,bbox=box,visible=visible))
                if visible:
                    cv2.circle(image,(x,y),15,(0,180,255),-1)
                    detections.append(box+[.9,32])
            tracked = tracker.update(Boxes(np.array(detections,dtype=np.float32).reshape(-1,6),(360,640)), image)
            predictions[f] = [dict(id=int(t[4]),bbox=t[:4].tolist(),confidence=float(t[5])) for t in tracked]
            truth.append(dict(frame=f,phase=phase,objects=objects))
            for p in predictions[f]:
                x1,y1,x2,y2 = map(int,p['bbox'])
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
                cv2.putText(image,str(p['id']),(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
            video.write(image)
    finally:
        video.release()
    gt = dict(frames=truth)
    (output/'gt.json').write_text(json.dumps(gt,indent=2),encoding='utf-8')
    report = evaluate(gt,predictions)
    report['mode'] = 'tracker_only_injected_detections'
    report['track_buffer'] = buffer
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'tracks.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame','id','x1','y1','x2','y2','confidence'])
        for frame, preds in predictions.items():
            for p in preds:
                writer.writerow([frame,p['id'],*p['bbox'],p['confidence']])
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/synthetic'))
    p.add_argument('--buffer',type=int,default=30)
    args=p.parse_args()
    print(json.dumps(generate(args.output,args.buffer),indent=2))


if __name__ == '__main__':
    main()
