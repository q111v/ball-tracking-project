"""Single basketball: specialized YOLO, moving image ROI, motion gating and built-in ByteTrack.
No video-frame coordinates, prerecorded tracks, or interpolation are used.
"""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


def head_regions(keypoints, person):
    face=np.asarray(keypoints)[:5]
    valid=face[face[:,2]>.3]
    if not len(valid):
        return []
    height=person[3]-person[1]
    return [(valid[:,:2].mean(axis=0)-np.array([0,height*.03]),max(11,height*.11))]


def association_box(box):
    box=np.asarray(box,dtype=float)
    center=(box[:2]+box[2:])/2
    half=max(128,float(np.max(box[2:]-box[:2]))/2)
    return np.r_[center-half,center+half]


class BasketballTracker:
    def __init__(self, model, device='cpu', roi_size=192):
        self.model=YOLO(str(model))
        self.pose=YOLO(str(Path(model).parent/'yolov8n-pose.pt'))
        self.heads=[]
        self.classes=[k for k,v in self.model.names.items() if v.lower() in ('ball','basketball','sports ball')]
        if not self.classes:
            raise ValueError('Model has no ball class')
        self.device=device
        self.roi_size=roi_size
        self.center=None
        self.velocity=np.zeros(2)
        self.misses=0
        self.template=None
        self.tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,
            new_track_thresh=.35,track_buffer=45,match_thresh=.99,fuse_score=False),frame_rate=30)

    def candidates(self,image,region,global_search=False):
        x1,y1,x2,y2=region
        crop=image[y1:y2,x1:x2]
        out=self.model.predict(crop,classes=self.classes,conf=.10,imgsz=1280 if global_search else 640,
                               device=self.device,verbose=False,iou=.4)[0]
        detections=out.boxes.data.cpu().numpy()
        candidates=[]
        for b in detections:
            box=b[:4]+np.array([x1,y1,x1,y1])
            w,h=box[2:]-box[:2]
            # Broadcast ball size bounds scale with source resolution, not frame number.
            size=max(image.shape[:2])/1280
            if min(w,h)<5*size or max(w,h)>65*size or not .45<w/h<2.2:
                continue
            candidates.append((box,float(b[4]),'yolo_roi' if not global_search else 'yolo_global'))
        return candidates

    def step(self,image):
        h,w=image.shape[:2]
        pose=self.pose.predict(image,conf=.25,imgsz=960,device=self.device,verbose=False)[0]
        self.heads=[]
        if pose.keypoints is not None:
            for keypoints, person in zip(pose.keypoints.data.cpu().numpy(),pose.boxes.xyxy.cpu().numpy()):
                self.heads.extend(head_regions(keypoints,person))
        global_search=self.center is None or self.misses>=20
        prediction=None if self.center is None else self.center+self.velocity
        if global_search:
            region=(0,0,w,h)
        else:
            radius=self.roi_size/2 + min(self.misses,5)*12
            region=(max(0,int(prediction[0]-radius)),max(0,int(prediction[1]-radius)),
                    min(w,int(prediction[0]+radius)),min(h,int(prediction[1]+radius)))
        candidates=self.candidates(image,region,global_search)
        scored=[]
        for box,conf,source in candidates:
            center=(box[:2]+box[2:])/2
            if any(np.linalg.norm(center-head)<radius for head,radius in self.heads):
                continue
            distance=0 if prediction is None else np.linalg.norm(center-prediction)
            if prediction is not None and distance>80+min(self.misses,8)*5:
                continue
            score=conf-distance/180
            if self.template is not None:
                a,b,c,d=box.astype(int)
                patch=image[max(0,b):min(h,d),max(0,a):min(w,c)]
                if patch.size:
                    patch=cv2.resize(patch,(24,24))
                    similarity=float(cv2.matchTemplate(patch,self.template,cv2.TM_CCOEFF_NORMED)[0,0])
                    score+=.25*similarity
            scored.append((score,box,conf,source))
        observation=None
        if scored:
            _,box,conf,source=max(scored,key=lambda x:x[0])
            center=(box[:2]+box[2:])/2
            if self.center is not None:
                self.velocity=.5*self.velocity+.5*(center-self.center)/max(1,self.misses+1)
            self.center=center
            self.misses=0
            a,b,c,d=box.astype(int)
            patch=image[max(0,b):min(h,d),max(0,a):min(w,c)]
            if self.template is None and patch.size:
                self.template=cv2.resize(patch,(24,24))

            observation=(box,conf,source)
        else:
            self.misses+=1
        # Expanded association windows help small fast balls whose physical boxes do not overlap.
        # Only actual YOLO boxes are rendered/exported, never these association windows.
        association=[]
        if observation:
            box,conf,_=observation
            association=[[*association_box(box),conf,0]]
        tracks=self.tracker.update(Boxes(np.asarray(association,np.float32).reshape(-1,6),image.shape[:2]),image)
        tid=int(tracks[0,4]) if len(tracks) else -1
        return observation,tid
