from ultralytics import YOLO


model = YOLO("path/to/best.engine")           # 载入已有的 TensorRT 引擎

results = model.predict(source="path/to/img",    # 可换为图片/文件夹/视频/摄像头(0)
                        imgsz=640,
                        conf=0.25,
                        device=0,             # 指定GPU
                        save=True,            # 保存可视化结果到 runs/predict/
                        save_txt=True,        # 保存 YOLO txt 标注
                        save_conf=True)       # txt中保存置信度
