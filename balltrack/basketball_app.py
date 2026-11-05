"""Basketball file processing with bounded lossless cache and optional backward pass."""
import csv
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from .basketball import BasketballTracker,assign_observation,new_tracker
from .history import Trails
from .temporal import select_trajectory,select_anchored_trajectory
from .runtime import configure_model

ROOT=Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def cache_write(path,image):
    ok,data=cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok:raise RuntimeError('Could not cache a video frame')
    Path(path).write_bytes(data.tobytes())


def cache_read(path):
    image=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise RuntimeError('Could not decode cached frame')
    return image


def chunk_capacity(shape,requested=120,budget_mb=384):
    return max(1,min(requested,int(budget_mb*1024*1024/max(1,np.prod(shape)))))


def run(args):
    if args.video is None or not Path(args.video).is_file():raise ValueError('Basketball mode requires an existing --video file')
    if args.threads<1 or args.max_frames<0 or args.trail<1:raise ValueError('Invalid thread/frame/trail arguments')
    if getattr(args,'select_ball',False) and args.no_display:raise ValueError('--select-ball requires a display window')
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    profile=getattr(args,'basketball_profile','auto')
    default_model='basketball_base.pt' if profile=='closeup' else 'basketball.pt'
    model=Path(getattr(args,'basketball_model',None) or ROOT/'models'/default_model)
    mode=getattr(args,'basketball_pass','bidirectional')
    detector=BasketballTracker(model,pose=YOLO(str(ROOT/'models/yolov8n-pose.pt')),
                                tracking=False,device=args.device,profile=profile)
    configure_model(detector.model,args.threads)
    configure_model(detector.pose,args.threads)
    cap=cv2.VideoCapture(str(args.video));writer=None;log=None;trace=None
    frames=observed=tracked=0;ids=set();last_id=None;histories=Trails(args.trail,45)
    try:
        if not cap.isOpened():raise RuntimeError('Cannot open video')
        fps=cap.get(cv2.CAP_PROP_FPS);fps=fps if 1<=fps<=240 else 30.
        identity_tracker=new_tracker(fps)
        log=(output/'tracks.csv').open('w',newline='',encoding='utf-8');table=csv.writer(log)
        table.writerow(['frame','time_s','id','x1','y1','x2','y2','confidence','source','pass','processing_s'])
        trace=(output/'observations.jsonl').open('w',encoding='utf-8')
        print('Basketball processing started. Bidirectional mode is offline; please wait for each chunk.',flush=True)
        manual_initialization=bool(getattr(args,'select_ball',False) or getattr(args,'initial_box',None))
        chosen_initial_box=None
        started=time.perf_counter();eof=False;read_frames=0
        while not eof and (not args.max_frames or read_frames<args.max_frames):
            with tempfile.TemporaryDirectory(prefix='.ball_frames_',dir=output) as temporary:
                paths=[];forward=[];durations=[];head_cache=[];limit=1 if mode=='forward' else 120
                for _ in range(limit):
                    if args.max_frames and read_frames>=args.max_frames:break
                    ok,image=cap.read()
                    if not ok:eof=True;break
                    if not paths:limit=1 if mode=='forward' else chunk_capacity(image.shape)
                    if read_frames==0 and (getattr(args,'select_ball',False) or getattr(args,'initial_box',None)):
                        if getattr(args,'initial_box',None):
                            x,y,w,h=args.initial_box
                        else:
                            print('Draw a rectangle around the basketball, then press ENTER. ESC cancels.',flush=True)
                            x,y,w,h=cv2.selectROI('Select basketball, ENTER confirms, ESC cancels',image.copy(),False,False)
                            cv2.destroyWindow('Select basketball, ENTER confirms, ESC cancels')
                        if w<=4 or h<=4 or x<0 or y<0 or x+w>image.shape[1] or y+h>image.shape[0]:
                            raise ValueError('Invalid or cancelled initial ball selection')
                        chosen_initial_box=list(map(float,[x,y,w,h]))
                        detector.seed(image,(np.array([x,y,x+w,y+h],float),1.,'manual_initialization'))
                    path=Path(temporary)/f'{len(paths):04d}.png';cache_write(path,image);paths.append(path)
                    tick=time.perf_counter();forward.append(detector.observe(image));head_cache.append(list(detector.heads));durations.append(time.perf_counter()-tick)
                    read_frames+=1
                    if read_frames%20==0:print(f"Forward detection: frame {read_frames}",flush=True)
                    if len(paths)>=limit:break
                if not paths:break
                backward=[None]*len(paths)
                if mode=='bidirectional':
                    last=next((i for i in range(len(paths)-1,-1,-1) if forward[i] is not None),None)
                    if last is not None:
                        reverse=BasketballTracker(detector.model,pose=detector.pose,tracking=False,device=args.device,profile=profile)
                        reverse.seed(cache_read(paths[last]),forward[last])
                        for i in range(last,-1,-1):
                            image=cache_read(paths[i]);tick=time.perf_counter();backward[i]=reverse.observe(image,heads=head_cache[i]);durations[i]+=time.perf_counter()-tick
                            if i%20==0:print(f"Backward detection: {i} frames left in chunk",flush=True)
                    shape=cache_read(paths[0]).shape
                    selected=(select_anchored_trajectory if manual_initialization else select_trajectory)(forward,backward,max(shape[:2])/1280)
                else:selected=forward
                def encode_observation(value):
                    return None if value is None else dict(bbox=list(map(float,value[0])),confidence=float(value[1]),source=value[2])
                for i,observation in enumerate(selected):
                    trace.write(json.dumps(dict(frame=frames,forward=encode_observation(forward[i]),backward=encode_observation(backward[i]),selected=encode_observation(observation)))+'\n')
                    image=cache_read(paths[i]);tick=time.perf_counter()
                    tid=assign_observation(identity_tracker,observation,image.shape[:2]);actual=[]
                    if observation is not None:
                        box,confidence,source=observation;observed+=1
                        center=tuple(map(int,(box[:2]+box[2:])/2))
                        if tid>=0:ids.add(tid);last_id=tid;tracked+=1;actual=[(tid,center)]
                        x1,y1,x2,y2=map(int,box)
                        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2);cv2.circle(image,center,3,(0,0,255),-1)
                        cv2.putText(image,f'Ball ID:{tid if tid>=0 else "?"} {confidence:.2f}',(max(0,x1-40),max(50,y1-10)),0,.6,(0,255,0),2)
                        origin='forward' if forward[i] is observation else 'backward'
                    else:
                        box=[0]*4;confidence=0.;source='missing';origin='none'
                        cv2.putText(image,f'ID:{last_id} NOT OBSERVED',(15,65),0,.7,(0,180,255),2)
                    histories.update(frames,actual)
                    for identity in list(histories.points):
                        for a,b in histories.segments(identity):cv2.line(image,a,b,(255,0,255),2)
                    rate=read_frames/max(time.perf_counter()-started,1e-9)
                    cv2.putText(image,f'Basketball | {mode} | process FPS:{rate:.1f}',(15,30),0,.6,(0,255,255),2)
                    if args.save:
                        if writer is None:
                            h,w=image.shape[:2];writer=cv2.VideoWriter(str(output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                            if not writer.isOpened():raise RuntimeError('Cannot write MP4; check output path and codec')
                        writer.write(image)
                    duration=durations[i]+time.perf_counter()-tick
                    table.writerow([frames,frames/fps,tid,*map(float,box),confidence,source,origin,duration]);frames+=1
                    if not args.no_display:
                        cv2.imshow('Basketball - Q/ESC to quit',image)
                        if cv2.waitKey(1)&255 in (ord('q'),27):eof=True;break
                print(f'Processed {frames} frames, {observed} observations',flush=True)
        if not frames:raise RuntimeError('No decoded frames')
        elapsed=time.perf_counter()-started
        summary=dict(frames=frames,observed_frames=observed,tracked_frames=tracked,observation_fraction=observed/frames,
            tracked_fraction=tracked/frames,unique_ids=sorted(ids),processing_fps=frames/elapsed,python=platform.python_version(),
            initialization="manual" if manual_initialization else "automatic",initial_box=chosen_initial_box,
            device=args.device,threads=args.threads,effective_threads=torch.get_num_threads(),mode=mode,profile=profile,model=str(model),detector_sha256=sha256_file(model),input_sha256=sha256_file(args.video),
            notes='Coverage is not accuracy. Missing frames have no invented boxes. Bidirectional mode is offline and may delay display by one chunk.')
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return summary
    finally:
        cap.release()
        if writer is not None:writer.release()
        if log is not None:log.close()
        if trace is not None:trace.close()
        if not args.no_display:cv2.destroyAllWindows()
