# 篮球视频测试操作指南

## 只需要使用两个入口

- `run_nba.cmd`：处理随包的原始 NBA 片段，不更换、不缩短原片。
- `篮球视频测试.cmd`：打开你自己的 MP4。选择“比赛转播”或“近景训练”；随后在第一帧用鼠标框住篮球，按 Enter 确认。此操作只指定初始目标，后续位置由检测与跟踪程序计算，不是手工绘制轨迹。

自动初始化也保留：`python run_basketball.py --video "D:\Videos\my_video.mp4" --auto`。多人、观众、图形标识容易使自动初始化产生歧义，首次测试新视频推荐框选模式。首帧须能看到球。

程序默认 CPU 离线双向处理：首次显示画面之前，需要处理一批帧，控制台会打印正向/反向进度。不是卡死，也不是保证实时。完成后在 `runs/basketball_时间/annotated.mp4` 查看原速视频。Q/ESC 提前退出；早退结果不代表完整视频通过。

## 参数

```bat
.conda-env\python.exe run_basketball.py --video "test.mp4" --profile broadcast
.conda-env\python.exe run_basketball.py --video "practice.mp4" --profile closeup
```

`--fast` 使用单向处理，速度较快但容易增加漏检；`--device 0` 在已安装匹配 CUDA 版 PyTorch 时使用 NVIDIA GPU。

已有初始框时，可用 `--initial-box X Y W H --no-display` 复现实验。该框只用于初始化，不写入输出真值，也不会在检测缺失时假装发现了球。

每次自动建立独立输出目录；指定 `--output` 时必须使用新的空目录。

## 如何理解画面

绿色框与红色中心点为有图像检测依据的球体位置；紫红色线为这些观察点的历史路径。遮挡或检测不足时显示 `NOT OBSERVED`，不伪造框。此时 ByteTrack 在缓存期内保留身份，重现后尝试恢复原 ID。长遮挡、突然切镜或目标出画后无法保证保持 ID。

## 可运行范围与精度边界

程序支持任意可由 OpenCV 解码的视频文件，但“文件能处理”不等于“每一帧都能准确检出球”。新视频建议采用清晰的720p/1080p画面，球至少十几像素，避免强模糊与长时间遮挡。可以使用自定义兼容YOLO权重 `--model "D:\Models\basketball.pt"`。

任何模型都不能预先保证未见过的所有篮球视频精准，包括极小球、严重遮挡、多个相似目标和大幅切镜。最终效果应查看输出视频并对照标注；误检、漏检、定位误差和ID切换需要分别评价，不把覆盖率当作准确率。
