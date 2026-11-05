# 第二版：短时遮挡预测（包含第一版实际跟踪修复）

合并更新包到原篮球专用项目，不删除原有 balltrack、models 文件夹；替换同名文件后在 PyCharm 运行 main.py。依赖和权重未增加，不用重装环境。

main.py：

```python
SCENE = "closeup"
PROCESSING_MODE = "auto"
PREDICT_OCCLUSION = True
PREDICTION_SECONDS = 0.3
```

紫色实线是实际观测轨迹；橙色虚线、空心点和 PREDICTED 标签表示短时估计。圆圈为经验不确定性半径，不是统计置信区间。

预测使用近期连续观测做图像坐标中的直线/二次运动拟合，并补偿相机运动。不使用未来帧补写“真实轨迹”，不向 YOLO 或 ByteTrack 注入虚构检测。默认最多预测 0.3 秒；历史不足、拟合不稳定、不确定性过大、超出画面或相机补偿失败时提前停止。重新检测后使用实测位置，ID 改变或预测与实测明显不一致时不强行连线。

tracks.csv 仍只记录实测框/漏检，预测另存 predictions.csv，source=predicted。summary.json 的 observed_frames、tracked_frames 不包含预测；predicted_frames 单独统计。可用 PREDICT_OCCLUSION=False 关闭。

碰地反弹、碰筐、传球、运动模糊、较长遮挡或视频跳剪都可能让估计错误。连续运动不意味着能唯一恢复隐藏轨迹。本输入约第 8 秒有明显人物姿态/位置跳变，更不能把消失前后的球位置直接当作同一段连续抛物线。
