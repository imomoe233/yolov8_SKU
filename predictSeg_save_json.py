import json
import os
import tqdm
from ultralytics import YOLO
from PIL import ImageOps,Image


model = YOLO(r"D:\code\yolov8\runs\Segment\shouyin_candy_huojia\weights\best.pt")
img_path = r'D:\code\datasets\auto_annotation\shouyin_candy\huojia_update'

for image_path_source in os.listdir(img_path):
    img_tmp = Image.open(os.path.join(img_path, image_path_source))
    # 使用exif_transpose来根据图像的EXIF信息调整图像方向
    transposed_image = ImageOps.exif_transpose(img_tmp)
    # 保存图像，设置高质量
    transposed_image.save(os.path.join(img_path, image_path_source), "JPEG", quality=98)
    #print("调整图片方向")


for file in os.listdir(img_path):
    if not (file.endswith('.jpg') or file.endswith('.png')):
        continue
    result = model(os.path.join(img_path, file))[0]
    orig_shape = result.orig_shape
    # 对于每一个json，都有这些数据
    data = {
    "version": "5.3.1",
    "flags": {},
    "imagePath": file,
    "imageData": None,
    "imageHeight": orig_shape[0],
    "imageWidth": orig_shape[1],
    "shapes": [],
    }
    
    # result.masks.shape[0] 查看有几个masks
    for i in range(len(result.masks)):
        name = result.names[int(result.boxes.cls[i].item())]
        point = result.masks.xy[i].tolist()
        # 每一个mask都有个shape，因此需要对于每个mask的shape都弄进去
        # 区别就是label和points不一样
        data['shapes'].append(
            {
                "label": name,
                "points": point,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            }
        )
    
    with open(os.path.join(img_path, file[:-4] + '.json'),'w') as f:
        json.dump(data, f, ensure_ascii=False)
    # 因为我希望每张图片都可以有一个单独的json文件，所以每次写完一个图片需要将jdict清空
    # 如果想将所有的内容写入一个json文件中(跟val一样),可以删除下面这一行代码
    
