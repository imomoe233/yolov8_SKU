from ultralytics import YOLO

from PIL import Image

# 加载YOLO模型
model = YOLO(r'D:\code\yolov8\runs\OBB\shibang_wcr\weights\best.pt')

# # 指定包含图片的输入目录
# input_dir = r"D:\code\datasets\aerbeisi\images\val"
# # 指定保存结果图片的输出目录
# output_dir = r'D:\code\yolov8\val\aerbeisi'

# 对图片运行推理
results = model(r'D:\code\datasets\shibang_wcr\images\val')

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image

# # 如果有检测到的边框，则绘制边框
# for result in results:
#     box = result.boxes  # 获取边框对象
#     print(result)