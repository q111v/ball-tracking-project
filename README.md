# 通用运动球检测与跟踪（PyCharm 版）

本分支为单独的通用版，与篮球专用版分开使用。官方 YOLOv8n COCO 权重检测 sports ball 类；Ultralytics 内置 ByteTrack 为多个球分配 ID。不区分篮球、足球、排球、网球等具体球种，不包含篮球专用模型、颜色筛选或针对 NBA 片段的微调。

## 运行

1. 解压后用 PyCharm 打开包含 main.py 的整个文件夹。
2. 使用之前安装好依赖的解释器，例如 E:\Anaconda\envs\PythonProject\python.exe。不要切回未安装依赖的 E:\python\python.exe。运行窗口第一行可确认实际解释器路径。
3. 修改 main.py 顶部 VIDEO_PATH，右键运行 main.py。

```python
VIDEO_PATH = r"C:\Users\QQQ\Desktop\足球测试.mp4"
```

依赖与篮球版相同，不需要重复安装。新环境才需要安装 requirements.txt。支持 Python 3.11、3.12。已附 models/yolov8n.pt，无需首次启动下载权重。

默认 videos/nba.mp4 是原来的 117 帧 NBA 短片，便于直观对比。videos/demo.mp4 是由 COCO 图片平移并添加遮挡生成的合成片段，不是实拍运动测试。来源见 videos 下的说明文件。

## 配置与输出

- IMAGE_SIZE=960：小球漏检可试 1280，但 CPU 变慢。不要因帧中偶然有框就认定跟踪稳定。
- CONFIDENCE=0.1：候选检测阈值，降低会增加误检。创建新 ID 还要求满足 configs/bytetrack.yaml 中的 new_track_thresh=0.25。
- DEVICE="cpu"：有兼容的 NVIDIA CUDA 环境才改为 "0"。
- CAMERA_COMPENSATION=True：估计背景平移、旋转和缩放，让历史轨迹随画面移动。
- TRAIL_SECONDS=0.7：仅显示最近 0.7 秒并淡出。
- TRAIL_SMOOTHING=0.8：仅平滑显示轨迹；1.0 关闭平滑。
- SHOW_WINDOW、SAVE_VIDEO：分别控制窗口和视频保存。窗口按 Q/ESC 结束。

每次结果自动保存在 results/时间戳/：annotated.mp4、tracks.csv、summary.json。CSV 保留原始框坐标，shot 与 id 共同标识轨迹；切镜头后不能仅凭相同数字 ID 判断为同一物理球。遮挡或漏检时不伪造球坐标，重新观测后断开旧线。

模型逐帧单向处理，FPS 为包括推理、运动估计和绘制的端到端吞吐；窗口打开时还包含显示耗时。不与篮球版双向离线处理直接作为同条件速度对比。

## 源码

- main.py：视频路径和参数。
- balltrack/general.py：模型.track(persist=True, tracker=...)、多球 ID、视频读写与 CSV。
- balltrack/motion.py：光流与 RANSAC 相机运动估计。
- balltrack/display_trail.py：短时渐隐、补偿和断线。
- balltrack/runtime.py：CPU 线程配置。
- configs/bytetrack.yaml：内置 ByteTrack 参数。
- tests/：自动化测试，开发验证时需 pytest。

## 局限与实验

sports ball 的训练覆盖面广，不代表能稳定检测任意球。远景足球、小球、模糊、遮挡、低清晰度转播都可能漏检；也可能把人头或其他圆形物误检为球。ByteTrack 依赖连续检测，漏检严重时无法修复，长遮挡后 ID 可能改变。该版不包含人体姿态排除模块。

二维背景补偿是近似，存在视差、镜头切换、纹理不足等限制；失败时清空旧轨迹。跟踪结果中的有框帧比例是覆盖率，不是检测准确率；没有人工真值不能宣称精准。本包的实测与失败结果见 VALIDATION.md。Windows 本地显示、摄像头和 GPU 未在云端验证；此简版仅提供本地视频入口。

本版模型没有使用示例视频训练。与见过 NBA 片段训练画面的篮球模型比较时，必须说明训练数据差异。推荐用独立视频人工标注中心点与可见性，分别统计误检、漏检、ID 切换、遮挡前后恢复与 FPS。

源码许可见 LICENSE。权重来源、哈希与 Ultralytics 授权说明见 models/SOURCE.json；第三方权重不由项目源码 MIT 许可证重新授权。
