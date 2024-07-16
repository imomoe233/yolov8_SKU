import os
from ultralytics import YOLO

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 少了个标签 正在重打

# Load a model
model = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\jd_chocolate_huojia.yaml',
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
            name='jd_chocolate_huojia',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)


# Load a model
model = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\jd_chocolate_cengban.yaml',
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
            name='jd_chocolate_cengban',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)


# Load a model
model = YOLO('D:\code\yolov8\save_model\guatiao_pretrained_best.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\jd_chocolate_shangpin.yaml',
            # pretrained=True,
            # model='D:\yolov8\guaTiao\\best.pt',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=16,
            close_mosaic=200, # mosaic data augmentation, close it by set the number same as epochs. Too low-level pixal with mosaic will make loss nan!
            workers=0,
            device='0',
            optimizer='auto',
            # resume=True,
            amp=False, # close amp
            # fraction=0.2,
            cos_lr=True,
            project='runs/OBB',
            name='jd_chocolate_shangpin',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)
