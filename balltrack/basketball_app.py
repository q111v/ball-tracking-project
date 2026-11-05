"""Video/UI/export layer for the explicit single-basketball broadcast mode."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import torch

from .basketball import BasketballTracker
from .history import Trails

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    if args.video is None or not Path(args.video).is_file():
        raise ValueError('Basketball mode requires an existing --video file')
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    model=ROOT/'models/basketball.pt'
    tracker=BasketballTracker(model,device=args.device)
    cap=cv2.VideoCapture(str(args.video))
    writer=None
    log=None
    histories=Trails(args.trail,45)
    frames,observed,tracked=0,0,0
    ids=set()
    seconds=0.
    last_id=None
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open video')
        fps=cap.get(cv2.CAP_PROP_FPS)
        fps=fps if 1<=fps<=240 else 30.
        log=(output/'tracks.csv').open('w',newline='',encoding='utf-8')
        table=csv.writer(log)
        table.writerow(['frame','id','x1','y1','x2','y2','confidence','source','processing_s'])
        while not args.max_frames or frames<args.max_frames:
            ok,image=cap.read()
            if not ok:
                break
            tick=time.perf_counter()
            observation,tid=tracker.step(image)
            actual=[]
            if observation is not None:
                box,confidence,source=observation
                observed+=1
                if tid>=0:
                    ids.add(tid);last_id=tid;tracked+=1
                    center=tuple(map(int,(box[:2]+box[2:])/2))
                    actual=[(tid,center)]
                else:
                    center=tuple(map(int,(box[:2]+box[2:])/2))
                x1,y1,x2,y2=map(int,box)
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(image,center,3,(0,0,255),-1)
                cv2.putText(image,f'Ball ID:{tid if tid>=0 else "?"} {confidence:.2f}',(max(0,x1-40),max(50,y1-10)),0,.6,(0,255,0),2)
            else:
                box=[0,0,0,0];confidence=0.;source='missing'
                cv2.putText(image,f'ID:{last_id} NOT OBSERVED',(15,65),0,.7,(0,180,255),2)
            histories.update(frames,actual)
            for identity in list(histories.points):
                for a,b in histories.segments(identity):
                    cv2.line(image,a,b,(255,0,255),2)
            rate=(frames+1)/max(seconds+time.perf_counter()-tick,1e-9)
            cv2.putText(image,f'NBA ball | FPS:{rate:.1f} | observed:{observed}/{frames+1}',(15,30),0,.65,(0,255,255),2)
            if args.save:
                if writer is None:
                    h,w=image.shape[:2]
                    writer=cv2.VideoWriter(str(output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot write video')
                writer.write(image)
            duration=time.perf_counter()-tick
            seconds+=duration
            table.writerow([frames,tid,*map(float,box),confidence,source,duration])
            frames+=1
            if not args.no_display:
                cv2.imshow('NBA Basketball - Q/ESC to quit',image)
                if cv2.waitKey(1)&255 in (ord('q'),27):
                    break
        if frames==0:
            raise RuntimeError('No decoded frames')
        summary=dict(frames=frames,observed_frames=observed,tracked_frames=tracked,
            observation_fraction=observed/frames,tracked_fraction=tracked/frames,unique_ids=sorted(ids),
            processing_fps=frames/seconds,python=platform.python_version(),device=args.device,
            mode='single_basketball_yolo_local_roi_pose_head_exclusion_bytetrack',
            detector_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),
            notes='Fractions describe output coverage, not accuracy. Missing frames are not interpolated.')
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
        return summary
    finally:
        cap.release()
        if writer is not None:writer.release()
        if log is not None:log.close()
        if not args.no_display:cv2.destroyAllWindows()
