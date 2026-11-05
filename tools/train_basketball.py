"""Optional reproducible domain fine-tuning; NOT needed to run the application."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]


def prepare(output):
    data=json.loads((ROOT/'training/annotations.json').read_text(encoding='utf-8'))
    records={x['frame']:x for x in data['records']}
    for split in ['train','val']:
        (output/'images'/split).mkdir(parents=True,exist_ok=True)
        (output/'labels'/split).mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/data['source']))
    for frame in range(max(records)+1):
        ok,image=cap.read()
        if not ok:raise RuntimeError('Training source video ended early')
        if frame not in records:continue
        rec=records[frame];split=rec['split'];box=None if rec['bbox'] is None else np.asarray(rec['bbox'])
        variants=[(f'f{frame:03d}',image,box)]
        if split=='train' and box is not None:
            cx,cy=map(int,(box[:2]+box[2:])/2)
            for side in [256,512]:
                x=max(0,min(image.shape[1]-side,cx-side//2));y=max(0,min(image.shape[0]-side,cy-side//2))
                variants.append((f'f{frame:03d}_crop{side}',image[y:y+side,x:x+side],box-np.array([x,y,x,y])))
        if split=='train' and frame in [1,4,7]:
            for x in [100,400,700,1000]:variants.append((f'f{frame:03d}_neg{x}',image[592:720,x:x+128],None))
        for name,im,b in variants:
            ok,encoded=cv2.imencode('.jpg',im)
            if not ok:raise RuntimeError('Could not encode training image')
            (output/'images'/split/(name+'.jpg')).write_bytes(encoded.tobytes())
            if b is None:label=''
            else:
                h,w=im.shape[:2];cx,cy=(b[:2]+b[2:])/2;bw,bh=b[2:]-b[:2]
                label=f'0 {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}\n'
            (output/'labels'/split/(name+'.txt')).write_text(label)
    cap.release()
    names=YOLO(str(ROOT/'models/basketball_base.pt')).names
    config=output/'dataset.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(output.resolve()),train='images/train',val='images/val',names=names)),encoding='utf-8')
    return config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'runs/domain_training')
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=15)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    torch.set_num_threads(2)
    config=prepare(args.output/'dataset')
    model=YOLO(str(ROOT/'models/basketball_base.pt'))
    model.train(data=str(config),epochs=args.epochs,imgsz=640,batch=4,device=args.device,workers=0,
        freeze=10,optimizer='AdamW',lr0=.0005,lrf=.2,warmup_epochs=1,patience=8,
        mosaic=0,mixup=0,scale=.15,translate=.05,fliplr=.5,hsv_h=.01,hsv_s=.3,hsv_v=.2,
        amp=False,plots=False,cache=False,project=str(args.output),name='training',exist_ok=True,
        seed=7,deterministic=True,save=True,verbose=False)


if __name__=='__main__':main()
