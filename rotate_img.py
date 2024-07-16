import os
from PIL import Image,ImageOps


pic_path = r'D:\code\datasets\auto_annotation\shouyin_candy\shangpin_update'


for image_path_source in os.listdir(pic_path):
    try:
        img_tmp = Image.open(os.path.join(pic_path, image_path_source))
        # 使用exif_transpose来根据图像的EXIF信息调整图像方向
        transposed_image = ImageOps.exif_transpose(img_tmp)
        # 保存图像，设置高质量
        transposed_image.save(pic_path, "JPEG", quality=98)
    except:
        pass