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

## 镜头移动时的轨迹显示（本次更新）

main.py 新增：

```python
CAMERA_COMPENSATION = True
TRAIL_SECONDS = 0.7
TRAIL_SMOOTHING = 0.8
```

背景特征点通过双向光流和 RANSAC 估计相邻帧的平移、旋转和缩放，把历史点对齐到当前画面。人体附近及画面顶部、底部的区域不参与估计。短轨迹随年龄淡出；漏检后重新出现、ID 变化、明显跳点、切镜头或运动估计失败时断开旧线。检测中心点、检测框和 CSV 坐标保持原始检测结果，只有轨迹线使用显示平滑。

这是近似二维补偿：观众与球场的视差、快速摇镜、极端缩放和动态遮挡仍可能导致估计失败。失败时宁可清空旧线，不能保证每一帧都有连续轨迹。运动状态数量与轨迹重置次数写入 summary.json，便于检查补偿是否经常失败。

只更新显示功能时替换 main.py、balltrack/video.py，并添加 balltrack/motion.py 与 balltrack/display_trail.py；不需要替换模型或重新安装依赖。替换 main.py 前保留自己的 VIDEO_PATH 配置。

## closeup 跟踪修复版

默认视频现为 videos/basketball_10s.mp4，SCENE="closeup"，PROCESSING_MODE="auto"。近景使用单向处理，转播使用原双向处理。新的 balltrack/closeup.py 保留全图语义上下文，排除篮圈/球网形状的重复球框，并在短时漏检时尝试局部放大补检。强全图观测可以重新捕获球，避免一直搜索错误的旧位置。近景不再逐帧运行人体姿态模型，broadcast 继续保留姿态排除逻辑。

第一阶段使用方法见 UPDATE_CLOSEUP.md，完整对比见 validation/closeup/。有球网排除与速度改进不代表所有篮球视频都能精准跟踪。

## 第二阶段：短时遮挡预测

当前分支启用 PREDICT_OCCLUSION=True，PREDICTION_SECONDS=0.3。实际观测轨迹为紫色实线，估计段为橙色虚线并显示 PREDICTED。漏检可能来自遮挡，也可能来自检测失败；程序不声称自动判断了遮挡的真实原因。

预测模块 balltrack/occlusion.py 使用近期连续观测做图像坐标中的直线/二次拟合；最多外推 0.3 秒，并在历史不足、不稳定、移出画面或相机补偿失败时停止。预测单独写入 predictions.csv，不计入检测/跟踪成功率，不作为 ByteTrack 的真实观测。误差圆是经验显示范围，不是校准后的置信区间。重新观测后以实测为准，不强行连接明显不符或不同 ID 的轨迹。

使用说明见 UPDATE_PREDICTION.md，预测效果与模拟遮挡验证见 validation/prediction/。切回 PREDICT_OCCLUSION=False 即回到仅实测显示，不需要更换模型。默认 closeup 单向处理可按当前及历史帧工作；broadcast 双向模式的检测本身使用离线信息，不应宣称整条处理流水线为实时因果预测。
