import csv, json, sys
from pathlib import Path
import numpy as np

checks = json.loads(Path(__file__).with_name("nba_checkpoints.json").read_text())
rows = {int(row["frame"]): row for row in csv.DictReader(Path(sys.argv[1]).open())}
hits = []
wrong = []
for check in checks["frames"]:
    row = rows[check["frame"]]
    center = np.array(
        [
            (float(row["x1"]) + float(row["x2"])) / 2,
            (float(row["y1"]) + float(row["y2"])) / 2,
        ]
    )
    hits.append(
        row["source"] != "missing"
        and np.linalg.norm(center - check["center"]) <= checks["tolerance_px"]
    )
for check in checks["negative_checks"]:
    row = rows[check["frame"]]
    x1, y1, x2, y2 = check["bbox"]
    center = np.array(
        [
            (float(row["x1"]) + float(row["x2"])) / 2,
            (float(row["y1"]) + float(row["y2"])) / 2,
        ]
    )
    if row["source"] != "missing" and x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
        wrong.append(check["frame"])
print(
    json.dumps(
        dict(
            visible_checks=len(hits),
            hits=sum(bool(x) for x in hits),
            negative_checks=len(checks["negative_checks"]),
            wrong_frames=wrong,
        ),
        indent=2,
    )
)
