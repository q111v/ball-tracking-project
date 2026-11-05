"""Run the validated close-up ball demonstration and audit its actual track output."""
import sys
from datetime import datetime

from balltrack.app import ROOT, parser, run
from tools.audit_track import audit


if __name__ == '__main__':
    output=ROOT/'runs'/('verified_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args=parser().parse_args(['--video',str(ROOT/'data/verified/ball_demo.mp4'),
                              '--model',str(ROOT/'yolov8s.pt'),'--color-assist','--save',
                              '--trail','180','--output',str(output),*sys.argv[1:]])
    run(args)
    report=audit(args.output)
    print('Track continuity audit:', 'PASS' if report['passed'] else 'FAIL')
    print('Output:', args.output)
    if not report['passed']:
        raise SystemExit(1)
