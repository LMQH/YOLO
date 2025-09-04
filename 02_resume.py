# 训练中断了可以继续训练
from ultralytics import YOLO


if __name__ == '__main__':
    # 之前训练的 last.pt 文件
    model = YOLO('runs/detect/train/weights/last.pt')

    model.train(resume=True)
