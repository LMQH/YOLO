# 导入 YOLO
from ultralytics import YOLO


# 把代码放在 main 方法中，否则会报错
if __name__ == '__main__':

    # 使用 yolo11s.pt 作为预训练模型，不做网络结构的任何更改
    model = YOLO('yolo11s.pt')
    # 训练模型 --- 这里的参数就特别多，无需每一个都记下来，只需要记住常用的基本命令就可以
    model.train(
        data="datasets/data.yaml",
        epochs=3,
        device=0,
        batch=8,
        workers=8
    )






