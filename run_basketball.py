"""Choose a basketball video and run without editing any source code."""
import argparse
import sys
from datetime import datetime

from balltrack.app import ROOT,parser,run


def choose_video(profile):
    try:
        import tkinter as tk
        from tkinter import filedialog,messagebox
        root=tk.Tk();root.withdraw()
        name=filedialog.askopenfilename(title='Choose basketball video',filetypes=[('Video','*.mp4 *.avi *.mov *.mkv'),('All files','*.*')])
        if name and profile is None:
            broadcast=messagebox.askyesno("视频类型", "这是篮球比赛转播或全场画面吗？\n是：比赛转播。否：近景训练。")
            profile="broadcast" if broadcast else "closeup"
        root.destroy()
        return name,profile
    except Exception as exc:
        raise RuntimeError('Cannot open file chooser; use --video "path/to/video.mp4"') from exc


if __name__=='__main__':
    options=argparse.ArgumentParser(description='Basketball video launcher')
    options.add_argument('--video',help='Omit to open a file chooser')
    options.add_argument('--profile',choices=['broadcast','closeup','auto'],default=None)
    options.add_argument('--fast',action='store_true',help='Forward pass only, lower accuracy')
    options.add_argument('--auto',action='store_true',help='Automatic initialization instead of first-frame selection')
    options.add_argument('--initial-box',type=float,nargs=4,metavar=('X','Y','W','H'))
    options.add_argument('--select-ball',action='store_true',help='Select initial ball when automatic initialization is ambiguous')
    options.add_argument('--threads',type=int,default=3)
    options.add_argument('--device',default='cpu')
    options.add_argument('--no-display',action='store_true')
    options.add_argument('--output')
    options.add_argument('--model',help='Optional compatible basketball YOLO weights')
    known=options.parse_args()
    if known.video:
        video,profile=known.video,known.profile or "broadcast"
    else:
        video,profile=choose_video(known.profile)
    if not video:sys.exit('No video selected')
    output=known.output or str(ROOT/'runs'/('basketball_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    values=['--video',video,'--basketball-mode','--basketball-profile',profile,'--save',
            '--threads',str(known.threads),'--trail','120','--device',known.device,'--output',output]
    if known.fast:values+=['--basketball-pass','forward']
    if known.initial_box:values+=['--initial-box',*map(str,known.initial_box)]
    elif known.select_ball or (not known.auto and not known.no_display):values+=['--select-ball']
    if known.no_display:values+=['--no-display']
    if known.model:values+=['--basketball-model',known.model]
    try:
        run(parser().parse_args(values))
        print('Saved results:',output)
    except (OSError,RuntimeError,ValueError) as exc:
        sys.exit(f'Error: {exc}')
