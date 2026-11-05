"""Evaluate explicit point checks and rejected-object checks; never call coverage accuracy."""
import argparse
import csv
import json
import math
from pathlib import Path


def evaluate(rows,annotations):
    by_frame={int(x['frame']):x for x in rows}
    checks=[]
    for truth in annotations['frames']:
        row=by_frame.get(truth['frame'])
        present=row is not None and row['source']!='missing'
        center=None if not present else [(float(row['x1'])+float(row['x2']))/2,(float(row['y1'])+float(row['y2']))/2]
        error=None if center is None else math.dist(center,truth['center'])
        checks.append(dict(frame=truth['frame'],error_px=error,hit=error is not None and error<=annotations.get('tolerance_px',12),
                           id=int(row['id']) if present else None))
    negative=[]
    for truth in annotations.get('negative_checks',[]):
        row=by_frame.get(truth['frame']);bad=False
        if row is not None and row['source']!='missing':
            center=[(float(row['x1'])+float(row['x2']))/2,(float(row['y1'])+float(row['y2']))/2]
            x1,y1,x2,y2=truth['bbox'];bad=x1<=center[0]<=x2 and y1<=center[1]<=y2
        negative.append(dict(frame=truth['frame'],object=truth['object'],wrong_object=bad))
    errors=[x['error_px'] for x in checks if x['error_px'] is not None]
    return dict(visible_center_checks=len(checks),visible_center_hits=sum(x['hit'] for x in checks),
                mean_center_error_px=sum(errors)/len(errors) if errors else None,
                maximum_center_error_px=max(errors) if errors else None,
                negative_checks=len(negative),wrong_object_hits=sum(x['wrong_object'] for x in negative),
                per_frame=checks,negative_details=negative,
                note='Sparse point/negative checks, not full-video recall, AP or a universal accuracy guarantee.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True);p.add_argument('--centers',type=Path,required=True)
    a=p.parse_args()
    with (a.run/'tracks.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    result=evaluate(rows,json.loads(a.centers.read_text(encoding='utf-8')))
    (a.run/'point_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
