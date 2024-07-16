from ultralytics import YOLO
import yaml

# D:\anaconda3\envs\yolov8\Lib\site-packages\ultralytics\nn\modules\conv.py
# ⬆️ line 40 激活函数

# Load a model
model = YOLO(r'D:\code\yolov8\save_model\guatiao_pretrained_best.pt')  # pretrained YOLOv8n model

model.train(data=r'D:\code\yolov8\cfg\shibang_wcr.yaml',
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
            name='shibang_wcr',
            save=True,
            save_period=5,
            val=True,
            conf=0.55,
)

'''
# Load a model
model = YOLO('D:\code\yolov8\save_model\guatiao_pretrained_best.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\kebike.yaml',
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
            name='kebike',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)

'''
'''
# Load a model
model = YOLO('D:\code\yolov8\save_model\guatiao_pretrained_best.pt')  # pretrained YOLOv8n model

model.train(data='D:\code\yolov8\cfg\\zhenbaozhu.yaml',
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
            name='zhenbaozhu',
            save=True,
            save_period=5,
            val=True,
            split='val',
            conf=0.55,
)

'''

'''
# Run batched inference on a list of images
results = model("D:\code\yolov8\guaTiao\\aerbeisi\images\\val")  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
'''