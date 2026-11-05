"""Run with: python validation/closeup/evaluate_checks.py PATH_TO_TRACKS_CSV"""

import csv, json, sys
from pathlib import Path
import numpy as np

checks = json.loads(Path(__file__).with_name("checkpoints.json").read_text())
rows = {int(r["frame"]): r for r in csv.DictReader(Path(sys.argv[1]).open())}
errors = []
misses = []
false_net = []


def center(row):
    return (
        np.array([float(row[k]) for k in ["x1", "y1"]])
        + np.array([float(row[k]) for k in ["x2", "y2"]])
    ) / 2


for frame, point in checks["visible_points"].items():
    row = rows[int(frame)]
    distance = float(np.linalg.norm(center(row) - point))
    if row["source"] != "missing" and distance <= checks["center_hit_radius_px"]:
        errors.append(distance)
    else:
        misses.append(int(frame))
for frame in checks["net_false_positive_checks"]:
    row = rows[frame]
    if (
        row["source"] != "missing"
        and np.linalg.norm(center(row) - checks["net_center"])
        <= checks["net_check_radius_px"]
    ):
        false_net.append(frame)
print(
    json.dumps(
        dict(
            visible_checks=len(checks["visible_points"]),
            hits=len(errors),
            missed_frames=misses,
            matched_mean_center_error_px=float(np.mean(errors)) if errors else None,
            net_checks=len(checks["net_false_positive_checks"]),
            false_net_frames=false_net,
        ),
        indent=2,
    )
)
