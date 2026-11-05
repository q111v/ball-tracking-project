# 轨迹显示更新方法

无需重装环境或重新下载模型。

1. 关闭正在运行的程序，记下旧 main.py 中的 VIDEO_PATH。
2. 将更新包中的 main.py 覆盖项目同名文件。
3. 将更新包中的 balltrack 文件夹内容复制到项目的 balltrack 文件夹，选择替换同名文件。不要删除原文件夹及其其他模块。
4. 打开 main.py，恢复自己的 VIDEO_PATH；原来的模型仍放在 models 文件夹。
5. 在 PyCharm 中运行 main.py，新的处理视频保存在 results 的新时间戳子文件夹，旧输出不会自动更新。

本次修改：

| 文件 | 修改内容 |
| --- | --- |
| main.py | 新增相机补偿开关、轨迹时长和显示平滑配置 |
| balltrack/video.py | 接入补偿和渐隐绘制，保存运动估计状态统计 |
| balltrack/motion.py | 新增背景光流和 RANSAC 相机运动估计 |
| balltrack/display_trail.py | 新增历史点变换、短时渐隐、断线及显示平滑 |

默认 CAMERA_COMPENSATION=True、TRAIL_SECONDS=0.7、TRAIL_SMOOTHING=0.8。

想短一些改 TRAIL_SECONDS=0.5；想关闭显示平滑改 TRAIL_SMOOTHING=1.0。原始检测框、中心点和 CSV 不做平滑。切镜头后关联状态重置，ID 不能跨镜头视为同一个物理对象。

补偿是二维近似；视差、运动模糊和背景特征不足时可能失败。失败就断开历史线，避免继续绘制错误连接。修改轨迹显示不等于提高检测模型精度。
