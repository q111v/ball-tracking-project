"""Audit continuity of one tracked object; these are output checks, not ground-truth accuracy."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def audit(folder, minimum_coverage=.9, maximum_gap=8):
    summary=json.loads((folder/'summary.json').read_text())
    with (folder/'tracks.csv').open(newline='',encoding='utf-8') as f:
        rows=[r for r in csv.DictReader(f) if int(r['id'])>=0]
    ids=sorted(set(int(r['id']) for r in rows))
    frames=int(summary['frames'])
    observed={int(r['frame']) for r in rows}
    gap=longest=0
    for frame in range(frames):
        gap=0 if frame in observed else gap+1
        longest=max(longest,gap)
    centers=[((float(r['x1'])+float(r['x2']))/2,(float(r['y1'])+float(r['y2']))/2) for r in rows]
    distance=float(np.linalg.norm(np.diff(np.array(centers),axis=0),axis=1).sum()) if len(centers)>1 else 0
    report=dict(frames=frames,frames_with_id=len(observed),id_coverage=len(observed)/frames if frames else 0,
                unique_ids=ids,longest_untracked_gap_frames=longest,observed_path_length_px=distance,
                processing_fps=summary['processing_fps'],
                criteria=dict(minimum_frames=90,minimum_id_coverage=minimum_coverage,maximum_gap_frames=maximum_gap,minimum_path_length_px=200),
                note='Output continuity audit only; not detection recall, IDF1 or proof of accuracy on other videos.')
    report['passed']=frames>=90 and len(ids)==1 and report['id_coverage']>=minimum_coverage and longest<=maximum_gap and distance>=200
    (folder/'track_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    report=audit(args.run)
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
