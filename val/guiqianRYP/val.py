import os
from ultralytics import YOLO

from PIL import Image, ImageOps


pic_path = r'D:\code\datasets\guiqianRYP\cengban\images\val'
# 加载YOLO模型
shangpin_model = YOLO(r'D:\code\yolov8_SKU\runs\Segment\guiqianRYP_cengban\weights\last.pt')

for image_path_source in os.listdir(pic_path):
    try:
        img_tmp = Image.open(os.path.join(pic_path, image_path_source))
        # 使用exif_transpose来根据图像的EXIF信息调整图像方向
        transposed_image = ImageOps.exif_transpose(img_tmp)
        # 保存图像，设置高质量
        transposed_image.save(os.path.join(pic_path, image_path_source), "JPEG", quality=98)
        #print("调整图片方向")
    except Exception as e:
        print(f"Error processing image {image_path_source}: {e}")
        continue

# 对图片运行推理
shangpin_results = shangpin_model(pic_path, iou=0.5)

for r in shangpin_results:
    im_array = r.plot(line_width=5, font_size=8)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

    # im.show()  # show image
    path_1 = r.path.split('/')[-1][:-4]
    im.save(f"{path_1}_results_last.jpg")
