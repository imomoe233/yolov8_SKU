import os
import cv2

from ultralytics import YOLO
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


font = ImageFont.truetype('C:/windows/fonts/Arial.ttf', size=150)

model = YOLO(r'D:\code\yolov8_SKU\runs\OBB\sanjia_02\weights\best.pt')
results = model(r'D:\code\datasets\sanjia_shenhe')
#results = model(r'D:\code\datasets\sanjia_shenhe', iou=0.25)

i = 0

for r in results:
    count = len(0 if r.obb is None else r.obb.cls)
    
    im_array = r.plot(line_width=5, font_size=8, labels=False)  # plot a BGR numpy array of predictions
    
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    
    add_number = ImageDraw.Draw(im)
    add_number.rectangle([24, 26, 200, 210], fill=(0, 0, 0, 155), outline=None, width=0)
    add_number.text((30,30), f"{count}", font = font, fill = 'white')
    # im.show()  # show image
    path_1 = r.path.split('/')[-1][:-4]
    im.save(f"{path_1}_results.jpg")
    i += 1