#-*-codeing:uft-8-*-
import glob
from PIL import Image, ImageOps
import cv2
import os
import shutil
from ultralytics import YOLO

'''
1、使用model_path中的模型对image_path中的图片进行推理
2、将推理结果result.obb.xywhr保存为.txt文件，于图片放置于同一路径
3、然后将txt文件转换为rolabelimg可以打开的xml文件，放置在图片同一路径（便于rolabelimg直接打开）
4、模型推理后以txt格式将识别出的每个object和坐标保存在folder_path，该txt可以直接作为yolo的label使用

需要修改的参数
model_path:模型路径
image_path：要推理的数据的文件夹路径
folder_path：txt保存路径
'''

CLASSES = ["bqklqq", "jxmcqkl", "xgjxqkl", "jxcqkl", "mcqklqq", "cmwqklqq"]
# 模型路径
model_path = r'D:\code\yolov8\runs\OBB\candy5_shangpin_few\weights\best.pt'
# 要推理的数据的文件夹路径
image_path = r'D:\code\datasets\auto_annotation\candy5\shangpin\no_annotation'
# 模型推理后以txt格式将识别出的每个object和坐标保存
folder_path = r'D:\code\datasets\auto_annotation\candy5\shangpin\no_annotation\txt'

for img in os.listdir(image_path):
    try:
        image_path_source = os.path.join(image_path, img)
        img_tmp = Image.open(image_path_source)
        # 使用exif_transpose来根据图像的EXIF信息调整图像方向
        transposed_image = ImageOps.exif_transpose(img_tmp)
        # 保存图像，设置高质量
        transposed_image.save(image_path_source, "JPEG", quality=98)
    except Exception as e:
        pass


# 加载YOLO模型
model = YOLO(model_path)

# 对图片运行推理
results = model(image_path)


for r in results:
    xywhr = r.obb.xywhr.tolist()
    with open(os.path.join(r.path[:-4] + '.txt'), 'w') as file:
        for i in range(r.obb.xywhr.size()[0]):
            line_1 = str(xywhr[i][0]) + ' ' + str(xywhr[i][1]) + ' ' + str(xywhr[i][2]) + ' ' + str(xywhr[i][3]) + ' ' + str(xywhr[i][4])
            cls = str(int(r.obb.cls[i].item()))
            file.write(f'{cls} ' + line_1 + '\n')


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

for root,dirs,files in os.walk(image_path):
    for file in files:
        if file.endswith((".jpg", ".bmp", ".png",".jpeg")):
            img_path = os.path.join(root, file)
            name = file.rsplit(".", 1)[0]
            txt_path = os.path.join(image_path, name+".txt")   
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
            xml_write(img, labels, image_path, file)

# 使用glob模块找到所有以.txt结尾的文件
txt_files = glob.glob(os.path.join(image_path, '*.txt'))

if not os.path.exists(folder_path):
    # 如果不存在则创建文件夹
    os.makedirs(folder_path)
            
for i in txt_files:
    shutil.copy(i, folder_path)
    os.remove(i)