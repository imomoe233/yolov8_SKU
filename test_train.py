from ultralytics import YOLO

model = YOLO('D:\code\yolov8_SKU\cfg\custom\yolov8-GAM_head.yaml')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8_SKU\cfg\zhuanan_small_shangpin.yaml',
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
            project='runs/test',
            name='',
            save=True,
            save_period=50,
            val=True,
            split='val',
            conf=0.55,
)