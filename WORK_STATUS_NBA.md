# NBA 篮球跟踪调试暂存（未最终验收）

用户要求暂停，恢复后继续针对原始117帧NBA视频调试，不更换视频、不伪造轨迹。

## 已修改
- balltrack/basketball.py：公开篮球专用YOLO权重、动态ROI、人体关键点排除人头、扩大ByteTrack内部关联窗口；输出仍使用真实检测框。
- balltrack/basketball_app.py、run_nba.py：独立NBA入口，main.py新增 --basketball-mode；缺失帧明确标记，不插值假装检测。
- models/basketball.pt、models/yolov8n-pose.pt、models/SOURCES.json：当前实际使用的公开权重和来源。
- data/nba/nba.mp4：完整原始117帧；visible_centers.json：仅7个可见帧人工中心抽样，不能等同全片真值。
- tests/test_basketball.py：人头排除和遮挡后关联回归测试。

## 已知限制
- 颜色/模板匹配补偿出现误跟人头，已从NBA实现移除。
- 尚不能承诺全片逐帧精准跟踪，存在较多漏检，必须完成位置核查再交付。
- 来源项目173MB专用权重受下载通道限制，尚未取得；目前使用另一份公开6MB篮球权重。
- 本次WIP未推送GitHub；远程仍是此前近景球演示版本。

## 运行与待办
1. 查看 runs/nba-final/summary.json 和 annotated.mp4，核对实际输出；最新代码有等价辅助函数整理，需最终复跑。
2. 查看 /workspace/work/pytest-nba312.log 和 pytest-nba311.log，修复测试失败。
3. 对照 visible_centers.json 计算误差，人工核查全部检出帧，严禁将ID覆盖率当精度。
4. 当前代码入口 python run_nba.py --no-display；输出目录不能已存在非空。
5. 完成验收报告、Windows入口和README后，再按此前授权更新仓库 q111v/ball-tracking-project。


## nba-baseline 实测记录
```json
{
  "frames": 117,
  "frames_with_detections": 25,
  "detection_frame_fraction": 0.21367521367521367,
  "unique_track_ids": 1,
  "inference_fps": 1.7238634157943735,
  "processing_fps": 1.693827064131185,
  "wall_fps": 1.6860341267018573,
  "source_fps": 30.0,
  "python": "3.12.14",
  "ultralytics": "8.3.228",
  "torch": "2.7.1+cpu",
  "opencv": "4.11.0",
  "platform": "Linux-6.18.57-x86_64-with-glibc2.41",
  "processor": "",
  "threads": 3,
  "packages": {
    "ultralytics": "8.3.228",
    "torch": "2.7.1+cpu",
    "torchvision": "0.22.1+cpu",
    "numpy": "2.2.6",
    "opencv-python": "4.11.0.86",
    "scipy": "1.15.3",
    "lap": "0.5.12"
  },
  "device": "cpu",
  "args": {
    "video": "data/nba/nba.mp4",
    "camera": null,
    "basketball_mode": false,
    "color_assist": false,
    "model": "yolov8s.pt",
    "device": "cpu",
    "imgsz": 1280,
    "conf": 0.1,
    "tracker": "/workspace/ball-tracking-project/configs/bytetrack.yaml",
    "output": "runs/nba-baseline",
    "save": true,
    "no_display": true,
    "max_frames": 0,
    "trail": 90,
    "threads": 3
  },
  "tracker_yaml": "tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.25\ntrack_buffer: 30\nmatch_thresh: 0.8\nfuse_score: true\n",
  "model_sha256": "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a",
  "color_assist_inserted_observations": 0
}
```


## nba-refined 实测记录
```json
{
  "frames": 117,
  "observed_frames": 41,
  "tracked_frames": 37,
  "observation_fraction": 0.3504273504273504,
  "tracked_fraction": 0.3162393162393162,
  "unique_ids": [
    1
  ],
  "processing_fps": 3.7429243133762364,
  "python": "3.12.14",
  "device": "cpu",
  "mode": "single_basketball_yolo_local_roi_pose_head_exclusion_bytetrack",
  "detector_sha256": "010376003d9c899ce49b71f8825253c5e2a32501bd11cfddc2e3e0018dab21f2",
  "input_sha256": "4b725473dfb15f27795d03683b6b35edd00472ffb86506799bd6c3d7f6415b13",
  "notes": "Fractions describe output coverage, not accuracy. Missing frames are not interpolated."
}
```


## nba-final 实测记录
```json
{
  "frames": 117,
  "observed_frames": 45,
  "tracked_frames": 43,
  "observation_fraction": 0.38461538461538464,
  "tracked_fraction": 0.36752136752136755,
  "unique_ids": [
    1
  ],
  "processing_fps": 2.3257017692068387,
  "python": "3.12.14",
  "device": "cpu",
  "mode": "single_basketball_yolo_local_roi_pose_head_exclusion_bytetrack",
  "detector_sha256": "010376003d9c899ce49b71f8825253c5e2a32501bd11cfddc2e3e0018dab21f2",
  "input_sha256": "4b725473dfb15f27795d03683b6b35edd00472ffb86506799bd6c3d7f6415b13",
  "notes": "Fractions describe output coverage, not accuracy. Missing frames are not interpolated."
}
```
