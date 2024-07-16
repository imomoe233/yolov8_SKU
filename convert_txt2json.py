import os
import json
from tqdm import tqdm



folder_path = r'C:\Users\Lenovo\Downloads\predict3\labels'  # 修改为你的标签文件夹路径
image_folder = r'C:\Users\Lenovo\Downloads\predict3\images'  # 修改为你的图片文件夹路径
output_folder = r'C:\Users\Lenovo\Downloads\predict3\labels_json'  # 修改为你的输出文件夹路径


def get_image_size(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def txt_to_json(txt_file, image_folder):
    image_file = os.path.join(image_folder, os.path.splitext(os.path.basename(txt_file))[0] + '.jpg')
    image_width, image_height = get_image_size(image_file)

    shapes = []
    with open(txt_file, 'r') as f:
        for line in f:
            coords = line.strip().split()
            if len(coords) > 1:  # 至少包含 x 和 y 两个坐标
                label = coords[0]
                points = [[float(coords[i]) * image_width, float(coords[i+1]) * image_height] for i in range(1, len(coords), 2)]
                shape = {"label": 'stone',
                         "points": points,
                         "group_id": None,
                         "shape_type": "polygon",
                         "flags": {}}
                shapes.append(shape)

    return {"version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_file),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width}

def convert_folder_to_json(folder_path, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            txt_file = os.path.join(folder_path, filename)
            json_output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.json')
            json_data = txt_to_json(txt_file, image_folder)
            with open(json_output_file, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

if __name__ == '__main__':
    convert_folder_to_json(folder_path, image_folder, output_folder)
