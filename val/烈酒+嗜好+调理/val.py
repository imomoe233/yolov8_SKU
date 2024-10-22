from ultralytics import YOLO

from PIL import Image

# 加载YOLO模型
model = YOLO(r'D:\code\yolov8_SKU\runs\OBB\drink_hobby_seasoning_shangpin2\weights\best.pt')
# model = YOLO(r'D:\code\yolov8_SKU\runs\Segment\drink_hobby_seasoning_huojia\weights\best.pt')
# # 指定包含图片的输入目录
# input_dir = r"D:\code\datasets\aerbeisi\images\val"
# # 指定保存结果图片的输出目录
# output_dir = r'D:\code\yolov8\val\aerbeisi'

# 对图片运行推理
results = model(r'D:\code\yolov8_SKU\val\烈酒+嗜好+调理',
                iou=0.5,
                #conf=0.3,
                )
'''
results = model(r'https://cstorecloud.oss-cn-shanghai.aliyuncs.com//cstore_ai/shelves_img/2024-07-01/c118107f-7dbe-40ab-a2bf-a97347affdba.jpg', iou=0.4)
'''

for r in results:
    im_array = r.plot(line_width=5, font_size=8)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image

# # 如果有检测到的边框，则绘制边框
# for result in results:
#     box = result.boxes  # 获取边框对象
#     print(result)