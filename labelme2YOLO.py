import shutil
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from tqdm.contrib import tzip
import json

# python train.py --data kll/lp.yaml --epochs 10 --weights ./yolov5s.pt  --batch-size 20 --device 0 --workers 1

class_names = ['cb', 'dgcb', 'hjgxcb']

data_path = 'D:\code\datasets\drink_hobby_seasoning\cengban'

                
def crpd2yolo(images_path: str, labels_path: str, save_labels_top: str):
    
    if not os.path.exists(save_labels_top):
        os.makedirs(save_labels_top)
    
    for f1, f2 in tzip(os.listdir(images_path), os.listdir(labels_path)):
        path_image = f'{images_path}/{f1}'
        path_label = f'{labels_path}/{f2}'
        
        im = cv2.imread(path_image)
        w , h = im.shape[1], im.shape[0]
        
        if(Path(path_image).stem == Path(path_label).stem):
            with open(path_label, 'r', encoding='utf-8') as f:
                coorList = f.read().split(" ")
                type = 0
                xyxy = [int(coorList[0]), int(coorList[1]), int(coorList[4]), int(coorList[5])]
            x,y,w,h = xyxy2xywh(xyxy, w, h)
            with open(f'{save_labels_top}/{Path(path_label).name}', 'w', encoding='utf-8') as f:
                s = f'{type} {x} {y} {w} {h}'
                f.write(s)
        
def xyxy2xywh(xyxy: list, im0_w, im0_h):
    """
    xyxy
    """
    x = (xyxy[0] + xyxy[2]) / 2
    y = (xyxy[1] + xyxy[3]) / 2
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]   

    # norm
    x = x / im0_w
    y = y / im0_h
    w = w / im0_w
    h = h / im0_h
    
    return x,y,w,h

def check_labels(path):
    """
    检测 lables 文件是否有负值
    """
    for f in tqdm(os.listdir(path)):
        file = f'{path}/{f}'
        with open(file, 'r') as f:
            line = f.read().split(" ")
            for coord in line[1:]:
                assert float(coord) > 0, f'Coord should less than 0, file is {file}'
                           
def labelme2yolo(json_path: str, save_path: str):
    
    files = os.listdir(json_path)
    for fn in files:
        if not fn.endswith('.json'): continue
        
        with open(os.path.join(json_path, fn), 'r') as f, open(os.path.join(save_path, Path(fn).stem + '.txt'), 'w') as f2:
            json_data = json.load(f)
            fn = json_data['imagePath']
            h, w = json_data['imageHeight'], json_data['imageWidth']
            
            for shape in json_data['shapes']:
                lable = shape['label']
                points = shape['points']
                
                points = np.array(points)
                # yolov5 格式 cx cy w h
                xmin, xmax = max(0, min(points[:, 0])), min(w, max(points[:, 0]))
                ymin, ymax = max(0, min(points[:, 1])), min(h, max(points[:, 1]))
            
                bbox = [xmin, ymin, xmax, ymax]
                lines = xyxy2xywh(bbox, w, h)
                
                f2.write(f'{str(class_names.index(lable))} {" ".join(str(l) for l in lines)} \n')
                
                
    
    pass

def dir_exists(folder_path):
        # 判断文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")
         
       
if __name__ == '__main__':
    
    
    # 获取文件夹中所有文件
    files = os.listdir(data_path)

    jpg_files = {os.path.splitext(file)[0] for file in files if file.endswith('.jpg')}
    json_files = {os.path.splitext(file)[0] for file in files if file.endswith('.json')}

    # 找到没有对应xml文件的jpg文件
    jpg_without_json = jpg_files - json_files

    # 删除没有对应xml文件的jpg文件
    for jpg in jpg_without_json:
        jpg_path = os.path.join(data_path, f'{jpg}.jpg')
        os.remove(jpg_path)
        print(f'Deleted {jpg_path}')
    
    json_path = os.path.join(data_path, 'json')
    dir_exists(json_path)
    
    labelme2yolo(data_path, json_path)

    labels_train_path = os.path.join(data_path, 'labels/train')
    labels_val_path = os.path.join(data_path, 'labels/val')
    
    dir_exists(labels_train_path)
    dir_exists(labels_val_path)
    
    # 获取out_path文件夹中的所有txt文件
    txt_files = os.listdir(json_path)

    # 按文件名排序以确保一致性
    txt_files.sort()

    # 将最后7个文件放到val文件夹中，其余的文件放到train文件夹中
    val_files = txt_files[-7:]
    train_files = txt_files[:-7]

    # 移动文件到对应的文件夹中
    print(f'Moved .txt from {json_path} to {labels_val_path}')
    for file in val_files:
        src_path = os.path.join(json_path, file)
        dst_path = os.path.join(labels_val_path, file)
        shutil.move(src_path, dst_path)
    
    print(f'Moved .txt from {src_path} to {dst_path}')
    for file in train_files:
        src_path = os.path.join(json_path, file)
        dst_path = os.path.join(labels_train_path, file)
        shutil.move(src_path, dst_path)

    images_train_path = os.path.join(data_path, 'images/train')
    images_val_path = os.path.join(data_path, 'images/val')
    
    dir_exists(images_train_path)
    dir_exists(images_val_path)    
    
    # 获取out_path文件夹中的所有txt文件
    jpg_files = [file for file in os.listdir(data_path) if file.endswith('.jpg')]

    # 按文件名排序以确保一致性
    jpg_files.sort()

    # 将最后7个文件放到val文件夹中，其余的文件放到train文件夹中
    val_files = jpg_files[-7:]
    train_files = jpg_files[:-7]

    # 移动文件到对应的文件夹中
    print(f'Moved .jpg from {data_path} to {images_val_path}')
    for file in val_files:
        src_path = os.path.join(data_path, file)
        dst_path = os.path.join(images_val_path, file)
        shutil.move(src_path, dst_path)
    
    print(f'Moved .jpg from {data_path} to {images_train_path}')
    for file in train_files:
        src_path = os.path.join(data_path, file)
        dst_path = os.path.join(images_train_path, file)
        shutil.move(src_path, dst_path)
    
    shutil.rmtree(json_path)
    