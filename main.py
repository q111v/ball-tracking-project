"""在 PyCharm 中修改下面的配置，然后右键运行本文件。"""

from datetime import datetime
from pathlib import Path

# 换视频只改这一行。Windows 路径前面的 r 请保留。
VIDEO_PATH = r"videos/nba.mp4"
# 例如：VIDEO_PATH = r"C:\Users\QQQ\Desktop\篮球测试.mp4"

DEVICE = "cpu"  # 有可用 NVIDIA CUDA 环境时可改成 "0"
SCENE = "broadcast"  # 比赛转播用 broadcast；近距离投篮用 closeup
SELECT_BALL = False  # True：在首帧框选篮球，回车确认
SAVE_VIDEO = True
SHOW_WINDOW = True
PROCESSING_MODE = "bidirectional"  # 双向处理较慢；forward 为单向处理
THREADS = 2


def main():
    from balltrack.video import process_video

    root = Path(__file__).resolve().parent
    video = Path(VIDEO_PATH)
    if not video.is_absolute():
        video = root / video
    if not video.is_file():
        raise FileNotFoundError(f"找不到视频，请修改 main.py 中的 VIDEO_PATH：{video}")

    # 每次运行单独保存结果，避免覆盖之前的实验。
    output = root / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = process_video(
        video,
        output,
        device=DEVICE,
        profile=SCENE,
        select_ball=SELECT_BALL,
        save=SAVE_VIDEO,
        show_window=SHOW_WINDOW,
        mode=PROCESSING_MODE,
        threads=THREADS,
    )
    print(f"处理完成，结果保存在：{output}")
    print(f"总帧数：{result['frames']}，处理 FPS：{result['processing_fps']:.2f}")


if __name__ == "__main__":
    main()
