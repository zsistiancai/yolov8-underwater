from ultralytics import YOLO

# model = YOLO('./weights/yolov8n.pt')

# model.predict(source='./bondagepics/213717ujw7qsejff6xmjqj.jpg', project='./bondagepreds', save=True)


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(data='custom_dataset_cfg/data.yaml', epochs=100, project='my_train_output', name='my_exp1', imgsz=640, device=0, amp=False)