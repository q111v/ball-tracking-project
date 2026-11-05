# 网上真实视频测试：当前版本尚未完全达到目标

测试日期：2026-11-05。此测试在 Linux 云端进行，不是在用户的 Windows 桌面上执行。

## 下载与输入

来源仓库：https://github.com/abdullahtarek/tennis_analysis

原始下载地址：https://raw.githubusercontent.com/abdullahtarek/tennis_analysis/main/input_videos/input_video.mp4

项目内位置：`data/online/tennis.mp4`，1920×1080、30 FPS、214 帧，约 7.13 秒。该文件是自然网球比赛视频，不是照片生成的演示，也未预先绘制目标检测框。来源、下载日期与 SHA256 见 `data/online/SOURCE.json`。原始视频保留其原有权利；这里只作教学功能验证。

## 两组完整视频实测

| YOLO11n 输入尺寸 | 处理帧数 | 有输出框的帧数 | 有效 ID 数 | 含绘制/保存 FPS | 保存视频重新解码 |
|---|---:|---:|---:|---:|---|
| 640（默认） | 214 | 0 | 0 | 5.74 | 214 帧，通过 |
| 1280 | 214 | 51 | 0 | 2.79 | 214 帧，通过 |

两组均使用 CPU、conf=0.1、默认 ByteTrack 配置、4 个 torch 线程。Python 3.12 下执行全部真实视频处理；Python 3.11 / 3.12 各重新通过 9 项自动化测试。没有 Windows 本机、摄像头、GPU 的新验证记录。

**结论：当前版本不能在此远景高速网球视频上实现稳定 ID 和连续运动轨迹，也未达到源视频 30 FPS 的实时吞吐率。** 文件输入、逐帧处理、输出日志和保存视频功能通过。短遮挡恢复与双球 ID 的合成测试仍通过，但不能替代真实运动中的验证。

1280 中的 51 个“有框帧”不是 51 次正确检测，也不是检测成功率。所有输出 ID 均为 -1，代表未形成确认的跟踪结果，而不是所有帧都属于同一个目标。没有有效 ID 时，程序不会伪造轨迹。

## 人工查看与稀疏探针

查看均匀抽帧后，在 30、60、90、150、180 帧可以明确辨认球体；120 帧视觉位置含糊，未纳入。粗略手工框保存在 `data/online/gt_sampled.json`。框含小球边缘和部分拖影，不是经过两人复核的严格基准标注。

在这 5 个点上，使用“预测中心距离人工框中心 ≤12 原图像素”的探索性探针：640 为 0/5，1280 为 3/5。稀疏样本不足以推断全片召回率。附带的 IoU≥0.5 评估对几像素的边界误差非常敏感；粗框与小预测框可能中心正确而 IoU 不过阈值，因此不把该项视作正式论文检测成绩。原始结果及探针逐帧距离见 `evidence/online-tennis*/online_check.json`。

正式论文需要完整、独立、严格复核的真值，尤其是遮挡前后帧的真实身份。此真实片段未标注遮挡事件，所以不报告它的遮挡恢复率。

## 为什么当前配置失败

1. 原画面为 1080p，但球通常仅几个到十几个像素。缩小到 640 后有效球体信息不足；运动模糊进一步削弱预训练 sports ball 检测。
2. 分辨率提高后，检测变多，但细小球的逐帧位移可超过自身宽度，框重叠很小；ByteTrack 的运动预测和关联不足以持续确认新轨迹。仅延长 lost 缓存无法修复从未形成的稳定关联。
3. COCO 通用模型与远景高速网球的领域存在差异。来源项目本身也采用专门微调的网球模型，不能据此假设通用模型效果相同。

下一步建议先用近景篮球/足球、较少模糊的视频完成基础课设演示，再采集远景高速球的独立数据；若正式实验必须覆盖后者，需要比较专用公开权重或微调、ROI/分块检测和更适合小目标快速位移的关联策略。上述增强尚未实现或验证，不能写成已完成成果。

## 复现

在项目根目录执行：

```bat
python main.py --video data/online/tennis.mp4 --no-display --save --output runs/online640_new
python main.py --video data/online/tennis.mp4 --imgsz 1280 --no-display --save --output runs/online1280_new
python -m balltrack.evaluate --gt data/online/gt_sampled.json --tracks runs/online1280_new/tracks.csv --timings runs/online1280_new/frames.csv --output runs/online1280_new/evaluation_sampled.json
```

原视频缺失时：`python -m tools.download_online_video`。下载工具核对已记录的 SHA256，避免静默使用变化的样本。

## Windows 桌面使用

当前会话不能写入用户本机桌面。压缩包包含原视频和两组处理视频，无需本地再次下载视频。

- `windows/extract_to_desktop.cmd`：弹出 ZIP 选择框，将选定项目压缩包解压到系统实际桌面（也适配重定向到 OneDrive 的桌面）下的新时间戳目录。该脚本仍需要先将 ZIP 与此脚本下载到本机，不会自行从云端获取项目 ZIP。
- `windows/setup_and_test.cmd`：在 Anaconda Prompt 中运行，创建项目内 `.conda-env` 的 Python 3.12 环境、安装 CPU 依赖、跑自动测试及完整默认视频测试。安装需要网络。该 Windows 脚本已提供，但云端没有执行它，不能标记为本地验收通过。

结果保存在新的 `runs/windows_online_*` 文件夹。PyCharm 可选择 `.conda-env/python.exe` 作为解释器。
