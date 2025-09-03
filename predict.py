from ultralytics import YOLO


def predict():
    # 设置检测资源 --- 一张图像
    source = "datasets/test/images/metal29_jpg.rf.5cadc019ff77d286f7a7b0e9affd13df.jpg"
    # 加载模型
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model.predict(
        source=source,  # 资源
        # save=True,  # 保存结果,默认保存在 runs/detect/目录下
    )

    # 打印结果   <class 'list'>  results 中的数据就是检测图片的数量
    # print("检测结果：\n", results, "\n", type(results), "\n", len(results))
    # 存储所有结果的列表
    all_results = []
    # 遍历结果
    """
        目标检测结果 result 中的属性分析：
            names 属性：是一个字典，字典的 key 就是类别信息配置的索引值，value 就是类别名称
            path 属性：是一个字符串，就是检测的图片的访问路径
            save_dir 属性：是一个字符串，是图像检测结果的保存路径，是一个相对路径
            speed 属性：是一个字典，内容就是前置处理时间、推理时间、后置处理时间，单位 ms
            probs 属性：目标检测模型的结果，这个属性为空，但是图像分类，这个内容就是存储的分类结果
            boxes 属性：目标检测模型的结果存储在这里，比如置信度、边界框坐标、类别信息等，但是分类模型，这里为 None
    """
    for result in results:
        # print(result)
        # 获取分类信息
        print("分类信息：\n", result.names)
        # 获取 boxes
        """
            boxes 中的属性：
                cls：类别的索引值
                conf：置信度信息
                data：每一个边界框的xyxy+conf+类别索引
                xyxy：坐标信息
        """
        # 遍历 boxes 属性
        for box in result.boxes:
            # print(box)
            # 获取每一个边界框的 xyxy + conf + cls名称
            cls = result.names[int(box.cls.item())]
            conf = round(box.conf.item(), 2)  # 四舍五入保留2位小数
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            print("类别信息：\n", cls)
            print("置信度信息：\n", conf)
            print("边界框坐标信息：\n", x_min, y_min, x_max, y_max)
            all_results.append([cls, conf, x_min, y_min, x_max, y_max])
    print("所有结果：\n", all_results)
    # 处理视频每一帧，然后把这一帧用来做检测，把检测结果绘制上去，显示结果


if __name__ == '__main__':
    predict()
