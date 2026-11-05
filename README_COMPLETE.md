# 篮球检测与运动轨迹跟踪：完整修复版

这是完整项目，包含全部 Python 模块、四个模型及两个示例视频。不要当成增量包，不需要再复制旧项目文件。兼容 Python 3.11、3.12。

## 在 PyCharm 打开

1. 解压后，在 PyCharm 选择「文件 → 打开」，选择包含 main.py 的 basketball-project 文件夹。
2. 使用已装好依赖的解释器，例如 E:\Anaconda\envs\PythonProject\python.exe。不要改回没有依赖的 E:\python\python.exe。
3. 右键这个新项目里的 main.py，选择运行。不要继续运行旧 basketball-prediction-update 目录下的 main.py。

已有依赖无需重装。只有全新 Python 环境才安装 requirements.txt。包内已包含模型，不需要启动时下载。

## 换测试视频

只改 main.py 顶部：

```python
VIDEO_PATH = r"C:\Users\QQQ\Desktop\篮球测试.mp4"
SCENE = "auto"
PROCESSING_MODE = "auto"
```

默认自动判断场景：近距离练习使用基于公开权重微调的篮球模型和人体附近补检；多人转播使用转播模型、人体姿态排除和双向处理。控制台会打印 Selected scene: closeup 或 broadcast。必要时可以手动指定 SCENE="closeup" 或 "broadcast"。自动判断基于开头少量帧，不能保证复杂混剪都判断正确。

默认视频 videos/basketball_10s.mp4 是 10 秒室内练习；videos/nba.mp4 是原 NBA 片段。两段源视频及权重均保留原有授权范围，来源见 videos 和 models 下的说明。

## 修复内容

- 近景和转播都排除人头核心区域，避免仅凭高置信度把人头选成球。
- 近景新权重针对部分可见的持球画面微调，并在低分或漏检时做另一尺度与人体附近复核。
- 错误候选离原运动位置很远时，需更高的检测置信度，减少跳到手臂或其他物体。
- 篮圈和球网语义排除及短时区域记忆。
- 预测落入头部、缺乏可靠历史或不确定性太大时停止显示。
- 完整打包所有本项目模块，包含 balltrack/head_memory.py，修复将增量包单独运行造成的缺文件问题。

## 显示与结果

绿色框和红色中心点是 YOLO 检测观测；紫色实线是实测轨迹；橙色虚线和 PREDICTED 是估计，默认最多 0.3 秒。预测不作为 ByteTrack 真实观测，不算检测成功率。PREDICT_OCCLUSION=False 可关闭预测。

每次结果在 results/时间戳/：annotated.mp4、tracks.csv、predictions.csv、observations.jsonl、summary.json。旧视频不会自动更新；请查看本次新生成的结果。转播双向处理需等待一批帧分析完成，不是实时播放；窗口按 Q/ESC 结束。

## 能力限制

遮挡、运动模糊、小球或球与篮筐重叠仍可能漏检；头部排除也可能保守排除紧贴脸部的真实球。长遮挡后 ID 可能改变。预测不能唯一恢复碰地、碰筐或传球后的隐藏运动，更不能跨视频跳剪保证连续。未检测到时宁可保留 missing，不把虚线称作真实检测。

程序不保证任意篮球视频都精准。稀疏检查点成绩不是整片准确率，完整运行与检查结果见 TEST_REPORT.md。broadcast 模型曾对随包 NBA 片段做过微调，该片段不能用于独立泛化结论；近景模型本轮使用这段视频的 24 个标注帧微调，并用同视频 6 个相邻帧选择权重；不是从零训练，也不能作为独立泛化证据。标注、参数及重现脚本随包附带。颜色恢复试验会误跟手臂，已弃用，交付版不使用该方法。Windows 显示、摄像头和 GPU 未在云端验证，此简版提供本地视频入口。

完整源码、自动测试和实验记录：
https://github.com/q111v/ball-tracking-project/tree/pycharm-prediction

源码许可见 LICENSE，模型和第三方素材仍遵守各自原有许可。
