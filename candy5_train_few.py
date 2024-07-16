import os
from ultralytics import YOLO


# D:\anaconda3\envs\yolov8\Lib\site-packages\ultralytics\nn\modules\conv.py
# ⬆️ line 40 激活函数


# Load a model
model = YOLO('D:\code\yolov8\save_model\guatiao_pretrained_best.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\candy5_shangpin_few.yaml',
            # pretrained=True,
            # model='D:\yolov8\guaTiao\\best.pt',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=4,
            close_mosaic=200, # mosaic data augmentation, close it by set the number same as epochs. Too low-level pixal with mosaic will make loss nan!
            workers=0,
            device='0',
            optimizer='auto',
            # resume=True,
            amp=False, # close amp
            # fraction=0.2,
            cos_lr=True,
            project='runs/OBB',
            name='candy5_shangpin_few',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)
