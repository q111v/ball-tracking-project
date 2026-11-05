# 篮球检测与轨迹跟踪：PyCharm 源码版

兼容 Python 3.11、3.12。直接运行 `main.py`，无需使用批处理启动脚本。

## 第一次打开

1. 解压后用 PyCharm 的「打开」选择本文件所在目录。
2. 在「设置 → 项目 → Python 解释器」选择之前已安装本项目依赖的解释器。不必重新下载 Anaconda 或重复安装整个环境。
3. 打开 `main.py`，右键选择「运行 main」。默认测试随包提供的 `videos/nba.mp4`。

如果提示 `No module named 'yaml'`，说明当前解释器缺少 PyYAML，或选错了解释器。先检查解释器是否为旧项目使用的那一个。确实缺少依赖时，可在 PyCharm 的解释器页面安装缺少的包（`yaml` 对应包名 `PyYAML`）。新环境则安装 `requirements.txt` 中的依赖。不要仅为这个报错重新安装 Anaconda。

## 测试自己的视频

只需修改 `main.py` 顶部这一行，再点击运行：

```python
VIDEO_PATH = r"C:\Users\QQQ\Desktop\篮球测试.mp4"
```

不要删除路径前的 `r`。程序以 main.py 所在目录解析相对路径，不依赖 PyCharm 工作目录。

比赛转播使用 `SCENE = "broadcast"`；近距离投篮使用 `"closeup"`。自动选错球时，可以设置 `SELECT_BALL = True`，在首帧框住篮球并按回车。这属于人工初始化，实验记录中应注明。

默认双向处理会先分析一批帧，再显示结果。CPU 上需要等待，并非实时播放；设置 `PROCESSING_MODE = "forward"` 可改为单向处理。窗口中按 Q 或 ESC 停止；双向分析期间不会立即响应窗口按键。

## 结果与源码

每次运行自动创建 `results/时间戳/`，包括处理视频 `annotated.mp4`、逐帧坐标 `tracks.csv`、摘要 `summary.json` 和观测记录。无需手动更换输出目录。

- `main.py`：修改视频路径、设备及显示设置，项目运行入口。
- `balltrack/video.py`：视频读取、双向检测、ByteTrack 关联、轨迹绘制与保存。
- `balltrack/basketball.py`：YOLO 篮球检测、局部搜索和外观筛选。
- `balltrack/head_memory.py`：利用人体姿态减少把头部识别成球。
- `balltrack/temporal.py`：选择时间连续的检测结果。
- `balltrack/history.py`、`runtime.py`：轨迹缓存与 CPU 线程设置。
- `models/`：已附所需权重；`videos/`：示例视频及来源。

使用 Ultralytics 的内置 BYTETracker，没有独立跟踪框架。本篮球版针对单个比赛用球；最初的通用多球项目及完整实验保留在仓库 main 分支。精简的是操作和文件布局，未声称重新提高算法精度。

## 能力边界与实验说明

通用 COCO sports ball 类覆盖多类运动球，但小目标、运动模糊、遮挡、球网重叠容易漏检。本版使用公开篮球预训练模型，并对随包 NBA 片段进行过微调，不能把该片段成绩当成独立泛化测试。来源和权重哈希见 models/SOURCES.json。

原片 117 帧中有 70 帧输出观测，8 个可见球人工检查点命中 7 个；另一个独立 NBA 片段仍跟踪不佳。因此不能保证任何篮球视频都精准。统计中的 observation_fraction 是有观测帧比例，不是准确率。遮挡和漏检帧不会伪造检测框，轨迹不会跨漏检帧连线；长遮挡后 ID 可能变化。

本版继续沿用已测试算法。源码版的 Python 3.11/3.12 验证结果见 VALIDATION.md。云端不验证 Windows 摄像头及桌面显示。

完整实验、失败案例及模型训练记录：
https://github.com/q111v/ball-tracking-project/blob/main/docs/NBA_RESULTS.md

本项目适合课程学习；请理解检测、关联和评估代码，并如实说明所用公开模型、修改内容及工具帮助。项目源码许可证见 LICENSE，第三方模型仍受其原许可证约束。
