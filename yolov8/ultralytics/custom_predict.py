from ultralytics import YOLO

model = YOLO('runs/detect/custom_run/exp1-3/weights/best.pt')

model.predict(source='../datasets/archive/aquarium_pretrain/test/images', project='./custom_preds/aquarium', save=True)