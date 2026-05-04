from ultralytics import YOLO

model = YOLO('./weights/yolov8n.pt')

model.train(data='custom_dataset_cfg/data.yaml', project='custom_run', name='exp1', epochs=50, batch=2,
            imgsz=640, save=True, device=0, workers=0, plots=True, cache=True, cos_lr=True, patience=10)