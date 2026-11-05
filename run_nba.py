"""One command to test the bundled NBA clip with the dedicated basketball mode."""
import sys
from datetime import datetime
from balltrack.app import ROOT,parser,run

if __name__=='__main__':
    args=parser().parse_args(['--video',str(ROOT/'data/nba/nba.mp4'),'--basketball-mode',
        '--save','--threads','3','--trail','117','--output',str(ROOT/'runs'/('nba_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))),*sys.argv[1:]])
    run(args)
