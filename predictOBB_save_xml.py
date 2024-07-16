import glob
import os
import shutil
from PIL import Image,ImageOps
import cv2
import numpy as np
import tqdm
from ultralytics import YOLO

'''
1、推理pic_path路径中的图片，识别货架的分割结果。
2、计算货架的外接矩阵，提取出商品的box。
3、如果有商品box的中心点在货架的外接矩阵中，则将该货架外接矩阵中所有的商品box坐标类别保存至txt。
4、然后将txt文件转换为rolabelimg可以打开的xml文件，放置在图片同一路径（便于rolabelimg直接打开）
5、模型推理后以txt格式将识别出的每个object和坐标保存在pic_path的txt中，该txt可以直接作为yolo的label使用

需要修改的参数
huojia_model: 识别货架的模型路径
shangpin_model: 识别商品的模型路径
pic_path: 要推理的数据的文件夹路径
CLASSES: 标签
'''

CLASSES = ["0"]
pic_path = r'D:\code\datasets\auto_annotation\shouyin_candy\shangpin_update'
huojia_model_path = r'D:\code\yolov8\runs\Segment\shouyin_candy_huojia\weights\best.pt'
# huojia_model_path = None
shangpin_model_path = r'D:\code\yolov8\runs\OBB\shouyin_candy_shangpin\weights\best.pt'


for image_path_source in os.listdir(pic_path):
    img_tmp = Image.open(os.path.join(pic_path, image_path_source))
    # 使用exif_transpose来根据图像的EXIF信息调整图像方向
    transposed_image = ImageOps.exif_transpose(img_tmp)
    # 保存图像，设置高质量
    transposed_image.save(os.path.join(pic_path, image_path_source), "JPEG", quality=98)
    #print("调整图片方向")

if huojia_model_path is not None:
    huojia_model = YOLO(huojia_model_path)
shangpin_model = YOLO(shangpin_model_path)

# huojia_results = huojia_model(r'D:\code\datasets\zhuanan_small\shangpin\images\val\f87e2929-c324-4608-866d-61718fe74a3e.jpg')
# shangpin_results = shangpin_model(r'D:\code\datasets\zhuanan_small\shangpin\images\val\f87e2929-c324-4608-866d-61718fe74a3e.jpg')

# 模型推理后以txt格式将识别出的每个object和坐标保存
folder_path = os.path.join(pic_path, 'txt')



def xml_write(rota_p_img, label, AUG_DIR, aug_name, mode = "robndbox"):
    xml_path = os.path.join(AUG_DIR,aug_name.rsplit(".", 1)[0]+".xml")
    image_path = os.path.join(AUG_DIR,aug_name)
    flag = 0
    for spt in label:
        if spt[4] in CLASSES:
            flag = 1
    
    if flag == 1:
        height, width = rota_p_img.shape[0:2]
        floder = AUG_DIR.split('\\')[-1]
        xml_file = open(xml_path, 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <source>\n')
        xml_file.write('        <database>Unknown</database>\n')
        xml_file.write('    </source>\n')
        xml_file.write('    <folder>' + floder + '</folder>\n')
        xml_file.write('    <filename>' + str(image_path) + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        for spt in label:
            if spt[4] not in CLASSES:
                continue
            xml_file.write('    <object>\n')
            xml_file.write('        <type>robndbox</type>\n')
            xml_file.write('        <name>' + spt[4]+ '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            if mode == "robndbox":
                xml_file.write('        <robndbox>\n')
                xml_file.write('            <cx>' + str(spt[0]) + '</cx>\n')
                xml_file.write('            <cy>' + str(spt[1]) + '</cy>\n')
                xml_file.write('            <w>' + str(spt[2]) + '</w>\n')
                xml_file.write('            <h>' + str(spt[3]) + '</h>\n')
                xml_file.write('            <angle>' + str(spt[5]) + '</angle>\n')
                xml_file.write('        </robndbox>\n')
            if mode == "bndbox":
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')

def point_in_polygon(point, polygon):
    # Use pointPolygonTest to check if the point is inside the polygon
    # It returns a positive value if the point is inside, negative if outside, and 0 if on the contour
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def calculate_bounding_box(segment_points):    
    # Initialize min and max values with the first point
    min_x, min_y = segment_points[0]
    max_x, max_y = segment_points[0]
    
    # Traverse all points to find the minimum and maximum coordinates
    for i in range(len(segment_points)):
        if segment_points[i][0] > max_x:
            max_x = segment_points[i][0]
        elif segment_points[i][0] < min_x:
            min_x = segment_points[i][0]
        elif segment_points[i][1] < min_y:
            min_y = segment_points[i][1]
        elif segment_points[i][1] > max_y:
            max_y = segment_points[i][1]
            
    # Define the four corner points of the bounding box
    top_left = (min_x, min_y)
    top_right = (max_x, min_y)
    bottom_left = (min_x, max_y)
    bottom_right = (max_x, max_y)
    
    return top_left, top_right, bottom_left, bottom_right



# if img.endswith('.jpg') or img.endswith('.png'):
if huojia_model_path is not None:
    huojia_results = huojia_model(pic_path)
shangpin_results = shangpin_model(pic_path)

for n in range(len(shangpin_results)):
    top_left_list = []
    bottom_right_list = []
    shangpin_n_cls = []
    cls_list = []
    if huojia_model_path is not None:
        for hj_result in huojia_results[n]:
            # hj_box = hj_result.boxes.xyxy.tolist()[0]
            for i in range(len(hj_result.masks.xy)):
                hj_mask = hj_result.masks.xy[i]
                # masks = [list(item) for item in masks]
                top_left, _, _, bottom_right = calculate_bounding_box(hj_mask)
                # print('boxbox!!' + hj_box)
                top_left_list.append(top_left)
                bottom_right_list.append(bottom_right)
            
    try:
        shangpin_xyxy = shangpin_results[n].obb.xyxy.tolist()
        shangpin_xywhr = shangpin_results[n].obb.xywhr.tolist()
    except:
        print("looks no identity in shangpin_results")
    result = []

    if huojia_model_path is not None:
        for i in range(len(shangpin_xyxy)):
            sp_box_temp = shangpin_xyxy[i]
            sp_box = [0, 0] # 保存box中心位置
            sp_box[1] = ((sp_box_temp[2] - sp_box_temp[0]) / 2.0) + sp_box_temp[0]
            sp_box[0] = ((sp_box_temp[3] - sp_box_temp[1]) / 2.0) + sp_box_temp[1]
            # print(sp_box)
            # result.append(shangpin_xywhr[i])
            
            for j in range(len(top_left_list)):
                top_left = top_left_list[j]
                bottom_right = bottom_right_list[j]
                # 如果当前商品中心点在货架外接矩阵中
                if top_left[1] < sp_box[0] < bottom_right[1] and top_left[0] < sp_box[1] < bottom_right[0]:
                    result.append(shangpin_xywhr[i])
                    #shangpin_n_cls.append(int(shangpin_results[n][i].obb.cls[0].tolist()))
                    cls_list.append(int(shangpin_results[n][i].obb.cls[0].tolist()))
                # 如果当前商品中心点不在货架外接矩阵中，则，下一个商品
                else:
                    continue
    else:
        for i in range(len(shangpin_xyxy)):
            result.append(shangpin_xywhr[i])
            #shangpin_n_cls.append(int(shangpin_results[n][i].obb.cls[0].tolist()))
            cls_list.append(int(shangpin_results[n][i].obb.cls[0].tolist()))
    
    with open(os.path.join(shangpin_results[n].path[:-4] + '.txt'), 'w') as file:
        for i in range(len(result)):
            line_1 = str(result[i][0]) + ' ' + str(result[i][1]) + ' ' + str(result[i][2]) + ' ' + str(result[i][3]) + ' ' + str(result[i][4])
            file.write(f'{cls_list[i]} ' + line_1 + '\n')


for root,dirs,files in os.walk(pic_path):
    for file in files:
        if file.endswith((".jpg", ".bmp", ".png",".jpeg")):
            img_path = os.path.join(root, file)
            name = file.rsplit(".", 1)[0]
            txt_path = os.path.join(pic_path, name+".txt")   
            if not os.path.exists(txt_path):
                continue
            img = cv2.imread(img_path)
            h,w = img.shape[0:2]
            labels = []
            with open(txt_path,'r',encoding='utf-8',errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    num = line.split(" ")
                    cls = CLASSES[int(num[0])]
                    cx = float(num[1])
                    cy = float(num[2])
                    cw = float(num[3])
                    ch = float(num[4])
                    angle = num[5]
                    labels.append([cx,cy,cw,ch,cls,angle])
            xml_write(img, labels, pic_path, file)
            
# 使用glob模块找到所有以.txt结尾的文件
txt_files = glob.glob(os.path.join(pic_path, '*.txt'))

if not os.path.exists(folder_path):
    # 如果不存在则创建文件夹
    os.makedirs(folder_path)
            
for i in txt_files:
    shutil.copy(i, folder_path)
    os.remove(i)