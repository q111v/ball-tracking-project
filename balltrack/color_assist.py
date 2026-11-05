"""Optional YOLO-seeded local color observations, inserted before built-in ByteTrack.

Designed for a single saturated-color ball against a distinct background. No box
is extrapolated when pixels do not support it. This is not a general ball detector.
"""
import cv2
import numpy as np
import torch


class ColorAssist:
    def __init__(self, class_id):
        self.class_id=class_id
        self.hue=None
        self.box=None
        self.inserted=0
        self.misses=0

    def __call__(self, predictor):
        for result in predictor.results:
            self.process(result)

    def process(self, result):
        image=result.orig_img
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        data=result.boxes.data if result.boxes is not None else np.empty((0,6),np.float32)
        rows=data.cpu().numpy() if isinstance(data,torch.Tensor) else np.asarray(data)
        if self.hue is None:
            seeds=[b for b in rows if int(b[5])==self.class_id and b[4]>=.35]
            if not seeds:
                return
            seed=max(seeds,key=lambda b:b[4])
            x1,y1,x2,y2=seed[:4]
            cx,cy=int((x1+x2)/2),int((y1+y2)/2)
            radius=max(2,int(min(x2-x1,y2-y1)*.25))
            patch=hsv[max(0,cy-radius):cy+radius+1,max(0,cx-radius):cx+radius+1]
            pixels=patch[(patch[:,:,1]>=80)&(patch[:,:,2]>=40)]
            if len(pixels)<20:
                return
            # Circular hue mode avoids a median discontinuity at red's 0/179 boundary.
            hist=np.bincount(pixels[:,0],minlength=180)
            self.hue=int(hist.argmax())
            self.box=seed[:4].copy()
        height,width=image.shape[:2]
        x1,y1,x2,y2=self.box
        bw,bh=x2-x1,y2-y1
        cx,cy=(x1+x2)/2,(y1+y2)/2
        left,top=max(0,int(cx-bw*1.6)),max(0,int(cy-bh*1.6))
        right,bottom=min(width,int(cx+bw*1.6)),min(height,int(cy+bh*1.6))
        crop=hsv[top:bottom,left:right]
        hue=crop[:,:,0].astype(np.int16)
        delta=np.abs(hue-self.hue)
        mask=((np.minimum(delta,180-delta)<=16)&(crop[:,:,1]>=75)&(crop[:,:,2]>=35)).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            x,y,w,h=cv2.boundingRect(contour)
            perimeter=cv2.arcLength(contour,True)
            if area<max(40,bw*bh*.16) or area>bw*bh*2.5 or perimeter<=0:
                continue
            if not .5<=w/h<=2 or 4*np.pi*area/(perimeter*perimeter)<.4:
                continue
            box=np.array([left+x,top+y,left+x+w,top+y+h],dtype=np.float32)
            distance=np.linalg.norm((box[:2]+box[2:])/2-np.array([cx,cy]))
            candidates.append((distance,box))
        if not candidates:
            self.misses+=1
            if self.misses>12:
                self.hue=self.box=None
            return
        self.misses=0
        box=min(candidates,key=lambda x:x[0])[1]
        self.box=box
        center=(box[:2]+box[2:])/2
        existing=[b for b in rows if int(b[5])==self.class_id and np.linalg.norm((b[:2]+b[2:4])/2-center)<min(box[2:]-box[:2])*.6]
        if existing:
            return
        # Low-confidence evidence may maintain a track but cannot create a new ID.
        observation=np.array([[*box,.22,self.class_id]],dtype=np.float32)
        combined=np.concatenate([rows,observation],axis=0)
        result.update(boxes=torch.as_tensor(combined,device=data.device if isinstance(data,torch.Tensor) else "cpu"))
        self.inserted+=1
