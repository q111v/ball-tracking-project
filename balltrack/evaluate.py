"""Evaluate explicitly annotated frames only. GT boxes use source pixel coordinates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    x = max(0, min(a[2], b[2])-max(a[0], b[0]))
    y = max(0, min(a[3], b[3])-max(a[1], b[1]))
    intersection = x*y
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection / union if union > 0 else 0.0


def evaluate(gt, predictions, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    groups = defaultdict(lambda: dict(tp=0, fp=0, fn=0, id_switches=0, matched_with_id=0))
    previous, recovered, gaps = {}, [], set()
    pending = {}
    seen_frames = set()
    for frame in sorted(gt['frames'], key=lambda f:f['frame']):
        index = int(frame['frame'])
        if index in seen_frames:
            raise ValueError('Duplicate annotated frame')
        seen_frames.add(index)
        objects = frame['objects']
        if len({str(o['id']) for o in objects}) != len(objects):
            raise ValueError('Duplicate GT identity in a frame')
        visible = [o for o in objects if o.get('visible', True)]
        for o in objects:
            if not o.get('visible', True):
                identity = str(o['id'])
                gaps.add(identity)
                pending.pop(identity, None)
        for o in visible:
            identity = str(o['id'])
            if identity in gaps:
                if identity in previous:
                    event = dict(before=previous[identity], after=None)
                    recovered.append(event)
                    pending[identity] = event
                gaps.remove(identity)
        pred = predictions.get(index, [])
        scores = np.array([[iou(o['bbox'], p['bbox']) for p in pred] for o in visible]).reshape(len(visible),len(pred))
        # Maximize valid match cardinality first, then total IoU.
        pairs = []
        if len(visible) and len(pred):
            rows, cols = linear_sum_assignment(-((scores >= threshold)*1000+scores))
            pairs = [(r,c) for r,c in zip(rows,cols) if scores[r,c] >= threshold]
        phase = frame.get('phase', 'all_frames')
        for key in set(('overall', phase)):
            group = groups[key]
            group['tp'] += len(pairs)
            group['fn'] += len(visible)-len(pairs)
            group['fp'] += len(pred)-len(pairs)
        for r,c in pairs:
            identity, tid = str(visible[r]['id']), int(pred[c]['id'])
            if tid < 0:
                continue
            switched = identity in previous and previous[identity] != tid
            for key in set(('overall', phase)):
                groups[key]['id_switches'] += int(switched)
                groups[key]['matched_with_id'] += 1
            if identity in pending:
                pending.pop(identity)['after'] = tid
            previous[identity] = tid
    for group in groups.values():
        tp,fp,fn = group['tp'],group['fp'],group['fn']
        group['precision'] = tp/(tp+fp) if tp+fp else None
        group['detection_success_rate_recall'] = tp/(tp+fn) if tp+fn else None
        group['tracked_match_fraction'] = group['matched_with_id']/tp if tp else None
        n = group['matched_with_id']
        group['id_stability_proxy'] = 1-group['id_switches']/n if n else None
    return dict(metrics=dict(groups), occlusion_recoveries_evaluated=len(recovered),
                same_id_after_occlusion=sum(e["after"] == e["before"] for e in recovered)/len(recovered) if recovered else None,
                occlusion_events_without_rematch=sum(e["after"] is None for e in recovered),
                note='Stability is a defined proxy, not IDF1/MOTA. Invisible objects excluded from recall; predictions remain FP.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', required=True, type=Path)
    p.add_argument('--tracks', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--timings', type=Path, help='Optional frames.csv for phase-specific processing FPS')
    p.add_argument('--warmup', type=int, default=5, help='Exclude first N frames from FPS only')
    p.add_argument('--iou', type=float, default=.5)
    args = p.parse_args()
    predictions = defaultdict(list)
    with args.tracks.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            predictions[int(row['frame'])].append(dict(id=int(row['id']), bbox=[float(row[k]) for k in ('x1','y1','x2','y2')]))
    gt = json.loads(args.gt.read_text(encoding='utf-8'))
    report = evaluate(gt, predictions, args.iou)
    if args.timings:
        phases = {int(f['frame']):f.get('phase','all_frames') for f in gt['frames']}
        durations = defaultdict(list)
        with args.timings.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                index = int(row['frame'])
                if index >= args.warmup and index in phases:
                    for key in set(('overall', phases[index])):
                        durations[key].append(float(row['processing_s']))
        report['processing_fps_by_phase'] = {k:len(v)/sum(v) for k,v in durations.items() if sum(v)>0}
        report['fps_warmup_frames'] = args.warmup
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
