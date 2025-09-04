from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')

    # 验证
    model.val(
        data='datasets/data.yaml',
        batch=16,
        # 启用混淆矩阵和PR曲线等图表
        plots=True
    )
