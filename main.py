"""在 PyCharm 中修改下面的配置，然后右键运行本文件。"""

from datetime import datetime
from pathlib import Path

# 换视频只改这一行。Windows 路径前面的 r 请保留。
VIDEO_PATH = r"videos/basketball_10s.mp4"
# 例如：VIDEO_PATH = r"C:\Users\QQQ\Desktop\篮球测试.mp4"

DEVICE = "cpu"  # 有可用 NVIDIA CUDA 环境时可改成 "0"
SCENE = "closeup"  # 比赛转播用 broadcast；近距离投篮用 closeup
SELECT_BALL = False  # True：在首帧框选篮球，回车确认
SAVE_VIDEO = True
SHOW_WINDOW = True
PROCESSING_MODE = "auto"  # auto：近景用单向处理；转播用双向处理
THREADS = 2

# 仅影响显示轨迹，CSV 仍保留原始检测坐标。
CAMERA_COMPENSATION = True  # 让历史轨迹随镜头平移、缩放
TRAIL_SECONDS = 0.7  # 只保留最近 0.7 秒，可改为 0.5 或 1.0
PREDICT_OCCLUSION = True  # 橙色虚线表示估计，紫色实线表示实际观测
PREDICTION_SECONDS = 0.3  # 最长预测 0.3 秒；更长时不再画预测位置
TRAIL_SMOOTHING = 0.8  # 新位置权重；1.0 表示不平滑


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
        camera_compensation=CAMERA_COMPENSATION,
        trail_seconds=TRAIL_SECONDS,
        trail_smoothing=TRAIL_SMOOTHING,
        predict_occlusion=PREDICT_OCCLUSION,
        prediction_seconds=PREDICTION_SECONDS,
    )
    print(f"处理完成，结果保存在：{output}")
    print(f"总帧数：{result['frames']}，处理 FPS：{result['processing_fps']:.2f}")


if __name__ == "__main__":
    main()
