# import tensorrt
# print(tensorrt.__version__)
from ultralytics import YOLO


# 加载训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 导出为 TensorRT 格式
model.export(format="engine", half=True)
