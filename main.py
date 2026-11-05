"""PyCharm 通用运动球检测入口：修改视频路径后直接运行。"""

from datetime import datetime
from pathlib import Path

VIDEO_PATH = r"videos/nba.mp4"
# VIDEO_PATH = r"C:\Users\QQQ\Desktop\足球.mp4"
MODEL_PATH = r"models/yolov8n.pt"  # 官方 COCO 通用权重，不是篮球专用模型
DEVICE = "cpu"
IMAGE_SIZE = 960  # 小球漏检可尝试 1280，CPU 会更慢
CONFIDENCE = 0.1  # 这是候选检测阈值；新 ID 还受跟踪阈值约束
CAMERA_COMPENSATION = True
TRAIL_SECONDS = 0.7
TRAIL_SMOOTHING = 0.8
SAVE_VIDEO = True
SHOW_WINDOW = True
THREADS = 2


def main():
    from balltrack.general import process_video

    root = Path(__file__).resolve().parent
    video = Path(VIDEO_PATH)
    model = Path(MODEL_PATH)
    if not video.is_absolute():
        video = root / video
    if not model.is_absolute():
        model = root / model
    output = root / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = process_video(
        video,
        output,
        model_path=model,
        device=DEVICE,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        camera_compensation=CAMERA_COMPENSATION,
        trail_seconds=TRAIL_SECONDS,
        trail_smoothing=TRAIL_SMOOTHING,
        save=SAVE_VIDEO,
        show_window=SHOW_WINDOW,
        threads=THREADS,
    )
    print(f"结果保存在：{output}")
    print(f"处理帧数：{result['frames']}，处理 FPS：{result['processing_fps']:.2f}")


if __name__ == "__main__":
    main()
