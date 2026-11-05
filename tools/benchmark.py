"""Repeat controlled end-to-end experiments; preserve every raw run."""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from balltrack.app import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/benchmark'))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--imgsz',type=int,nargs='+',default=[320,640])
    p.add_argument('--buffers',type=int,nargs='+',default=[2,30])
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.repeats<1 or any(n<0 for n in args.buffers):
        p.error('repeats must be >=1 and buffers >=0')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for size in args.imgsz:
        for buffer in args.buffers:
            cfg=yaml.safe_load((ROOT/'configs/bytetrack.yaml').read_text())
            cfg['track_buffer']=buffer
            config=args.output/f'buffer{buffer}.yaml'
            config.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            group=[]
            for repeat in range(args.repeats):
                output=args.output/f'size{size}_buffer{buffer}_repeat{repeat+1}'
                subprocess.run([sys.executable,str(ROOT/'main.py'),'--video',str(args.video.resolve()),
                                '--tracker',str(config),'--device',args.device,'--imgsz',str(size),
                                '--no-display','--output',str(output)],cwd=ROOT,check=True)
                subprocess.run([sys.executable,'-m','balltrack.evaluate','--gt',str(args.gt.resolve()),
                                '--tracks',str(output/'tracks.csv'),'--timings',str(output/'frames.csv'),
                                '--output',str(output/'evaluation.json')],cwd=ROOT,check=True)
                report=json.loads((output/'evaluation.json').read_text())
                group.append(report)
            fps=[r['processing_fps_by_phase']['overall'] for r in group]
            records.append(dict(imgsz=size,buffer=buffer,repeats=args.repeats,
                                processing_fps_mean=statistics.mean(fps),
                                processing_fps_std=statistics.stdev(fps) if len(fps)>1 else 0,
                                recall_mean=statistics.mean(r['metrics']['overall']['detection_success_rate_recall'] for r in group),
                                id_switches_mean=statistics.mean(r['metrics']['overall']['id_switches'] for r in group)))
    with (args.output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__=='__main__':
    main()
