"""Make a deterministic photo-translation clip from a local COCO128 dataset."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--coco128',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/demo'))
    args=p.parse_args()
    source=args.coco128/'images/train2017/000000000368.jpg'
    image=cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(source)
    h,w=image.shape[:2]
    labels=(args.coco128/'labels/train2017/000000000368.txt').read_text().splitlines()
    balls=[]
    for row in labels:
        cls,x,y,bw,bh=map(float,row.split())
        if cls==32:
            balls.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    args.output.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(args.output/'input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    frames=[]
    try:
        for f in range(90):
            dx=int(round(12*np.sin(f/20)))
            frame=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,0]]),(w,h),borderMode=cv2.BORDER_REFLECT)
            visible=not 35<=f<43
            objects=[]
            for i,base in enumerate(balls,1):
                box=[base[0]+dx,base[1],base[2]+dx,base[3]]
                objects.append(dict(id=i,bbox=box,visible=visible))
                if not visible:
                    x1,y1,x2,y2=map(int,box)
                    cv2.rectangle(frame,(x1-6,y1-6),(x2+6,y2+6),(25,25,25),-1)
            writer.write(frame)
            frames.append(dict(frame=f,phase='before' if f<35 else 'occluded' if f<43 else 'after',objects=objects))
    finally:
        writer.release()
    (args.output/'gt.json').write_text(json.dumps(dict(frames=frames),indent=2),encoding='utf-8')
    (args.output/'SOURCE.txt').write_text('Derived demo: translated COCO image 000000000368, with an artificial 8-frame occluder. Not a real motion benchmark.\nDataset source: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip\nCOCO source and image-specific licensing: https://cocodataset.org/#termsofuse\n',encoding='utf-8')


if __name__=='__main__':
    main()
