import os
from ultralytics import YOLO


# 少了个标签 正在重打
'''
# Load a model
model = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\drink_hobby_seasoning_huojia.yaml',
            # pretrained=True,
            # model='D:\yolov8\guaTiao\\best.pt',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=32,
            close_mosaic=200, # mosaic data augmentation, close it by set the number same as epochs. Too low-level pixal with mosaic will make loss nan!
            workers=0,
            device='0',
            optimizer='auto',
            # resume=True,
            amp=False, # close amp
            # fraction=0.2,
            cos_lr=True,
            project='runs/Segment',
            name='drink_hobby_seasoning_huojia',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)


# Load a model
model = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\drink_hobby_seasoning_cengban.yaml',
            # pretrained=True,
            # model='D:\yolov8\guaTiao\\best.pt',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=32,
            close_mosaic=200, # mosaic data augmentation, close it by set the number same as epochs. Too low-level pixal with mosaic will make loss nan!
            workers=0,
            device='0',
            optimizer='auto',
            # resume=True,
            amp=False, # close amp
            # fraction=0.2,
            cos_lr=True,
            project='runs/Segment',
            name='drink_hobby_seasoning_cengban',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)

'''

# Load a model
model = YOLO('D:\code\yolov8_SKU\cfg\custom\yolov8-CA_head_minDetectHead.yaml')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8_SKU\cfg\drink_hobby_seasoning_shangpin.yaml',
            # pretrained=True,
            # model='D:\yolov8\guaTiao\\best.pt',
            cache=True,
            imgsz=640,
            epochs=800,
            batch=16,
            close_mosaic=100, # mosaic data augmentation, close it by set the number same as epochs. Too low-level pixal with mosaic will make loss nan!
            workers=0,
            device='0',
            optimizer='auto',
            # resume=True,
            amp=False, # close amp
            # fraction=0.2,
            cos_lr=True,
            project='runs/OBB',
            name='drink_hobby_seasoning_shangpin_CA+minDH',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)
