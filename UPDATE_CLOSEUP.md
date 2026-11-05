# 第一版：先改好近景篮球实际跟踪（不含遮挡预测）

保留原环境和 models 文件夹，不必重新安装依赖。

将更新包内容复制到原篮球专用项目，合并 balltrack 和 videos 文件夹、替换同名文件，不要删除原文件夹中的其他模块。

- main.py：默认 closeup，PROCESSING_MODE="auto"，近景自动使用单向处理。
- balltrack/basketball.py：仅 closeup 接入新的检测流程，broadcast 保留原逻辑。
- balltrack/closeup.py：新增全图检测、篮圈/球网语义排除、背景补偿后的短时区域记忆、局部补检与重新捕获。
- balltrack/video.py：识别 auto 模式。
- videos/basketball_10s.mp4：同一份上传的 10 秒输入，未改变内容。

换自己的视频仍只改 main.py 的 VIDEO_PATH；近景保持 SCENE="closeup"，比赛转播改成 "broadcast"。

第一版没有给漏检帧画预测框或预测中心。依赖、权重和轨迹补偿不变。丢失过久或目标大幅位移后 ID 仍可能变化。

结果：完整 300 帧验证，19 个人工核对的清晰可见检查点从旧版命中 8 个提高到 19 个；25 个球网误检检查点从误检 7 个降为 0 个。这些为候选辅助定位、非随机的稀疏排错检查，不代表整片准确率或其他视频泛化能力。详见 validation/closeup/。
