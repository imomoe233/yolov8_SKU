# 功能描述   ：把rolabelimg标注的xml文件转换成dota能识别的xml文件，
#             再转换成dota格式的txt文件
#            把旋转框 cx,cy,w,h,angle，或者矩形框cx,cy,w,h,转换成四点坐标x1,y1,x2,y2,x3,y3,x4,y4
import os
import shutil
import xml.etree.ElementTree as ET
import math
from pathlib import Path
import os
import cv2
from ultralytics.utils import LOGGER, TQDM

# 修改为自己的标签
cls_list = ["0"]
# 修改为需要调整的路径，在这个路径下，要求包含.jpg和图片对应的.xml,需要是相同名字，例如A.xml和A.jpg
data_path = r'D:\code\datasets\guiqianRYP\shangpin'

# 需要调整标签对应的映射
class_mapping = {
    "0": 0,
}

def convert_dota_to_yolo_obb(dota_root_path: str):

    dota_root_path = Path(dota_root_path)

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir, image_path):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        try:
            with orig_label_path.open("r") as f, save_path.open("w") as g:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    class_name = parts[8]
                    class_idx = class_mapping[class_name]
                    coords = [float(p) for p in parts[:8]]
                    normalized_coords = [
                        coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                    ]
                    formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                    g.write(f"{class_idx} {' '.join(formatted_coords)}\n")
        except FileNotFoundError:
            os.remove(image_path)
            print(f"DELETE!!!{image_path}")

    for phase in ["train", "val"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            # if image_path.suffix != ".jpg":
            #     continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir, image_path)
 
def edit_xml(xml_file, dotaxml_file):
    """
    修改xml文件
    :param xml_file:xml文件的路径
    :return:
    """
 
    # dxml_file = open(xml_file,encoding='gbk')
    # tree = ET.parse(dxml_file).getroot()
 
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        x0 = ET.Element("x0")  # 创建节点
        y0 = ET.Element("y0")
        x1 = ET.Element("x1")
        y1 = ET.Element("y1")
        x2 = ET.Element("x2")
        y2 = ET.Element("y2")
        x3 = ET.Element("x3")
        y3 = ET.Element("y3")
        # obj_type = obj.find('bndbox')
        # type = obj_type.text
        # print(xml_file)
 
        if (obj.find('robndbox') == None):
            obj_bnd = obj.find('bndbox')
            obj_xmin = obj_bnd.find('xmin')
            obj_ymin = obj_bnd.find('ymin')
            obj_xmax = obj_bnd.find('xmax')
            obj_ymax = obj_bnd.find('ymax')
            # 以防有负值坐标
            xmin = max(float(obj_xmin.text), 0)
            ymin = max(float(obj_ymin.text), 0)
            xmax = max(float(obj_xmax.text), 0)
            ymax = max(float(obj_ymax.text), 0)
            obj_bnd.remove(obj_xmin)  # 删除节点
            obj_bnd.remove(obj_ymin)
            obj_bnd.remove(obj_xmax)
            obj_bnd.remove(obj_ymax)
            x0.text = str(xmin)
            y0.text = str(ymax)
            x1.text = str(xmax)
            y1.text = str(ymax)
            x2.text = str(xmax)
            y2.text = str(ymin)
            x3.text = str(xmin)
            y3.text = str(ymin)
        else:
            obj_bnd = obj.find('robndbox')
            obj_bnd.tag = 'bndbox'  # 修改节点名
            obj_cx = obj_bnd.find('cx')
            obj_cy = obj_bnd.find('cy')
            obj_w = obj_bnd.find('w')
            obj_h = obj_bnd.find('h')
            obj_angle = obj_bnd.find('angle')
            cx = float(obj_cx.text)
            cy = float(obj_cy.text)
            w = float(obj_w.text)
            h = float(obj_h.text)
            angle = float(obj_angle.text)
            obj_bnd.remove(obj_cx)  # 删除节点
            obj_bnd.remove(obj_cy)
            obj_bnd.remove(obj_w)
            obj_bnd.remove(obj_h)
            obj_bnd.remove(obj_angle)
 
            x0.text, y0.text = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
            x1.text, y1.text = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
            x2.text, y2.text = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
            x3.text, y3.text = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
 
        # obj.remove(obj_type)  # 删除节点
        obj_bnd.append(x0)  # 新增节点
        obj_bnd.append(y0)
        obj_bnd.append(x1)
        obj_bnd.append(y1)
        obj_bnd.append(x2)
        obj_bnd.append(y2)
        obj_bnd.append(x3)
        obj_bnd.append(y3)
 
        tree.write(dotaxml_file, method='xml', encoding='utf-8')  # 更新xml文件

# 转换成四点坐标
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc;
    yoff = yp - yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))
 
def totxt(xml_path, out_path):
    # 想要生成的txt文件保存的路径，这里可以自己修改
 
    files = os.listdir(xml_path)
    i = 0
    for file in files:
 
        tree = ET.parse(xml_path + os.sep + file)
        root = tree.getroot()
 
        name = file.split('.')[0]
 
        output = out_path + '\\' + name + '.txt'
        file = open(output, 'w', encoding='utf-8')
        i = i + 1
        objs = tree.findall('object')
        for obj in objs:
            cls = obj.find('name').text
            box = obj.find('bndbox')
            x0 = int(float(box.find('x0').text))
            y0 = int(float(box.find('y0').text))
            x1 = int(float(box.find('x1').text))
            y1 = int(float(box.find('y1').text))
            x2 = int(float(box.find('x2').text))
            y2 = int(float(box.find('y2').text))
            x3 = int(float(box.find('x3').text))
            y3 = int(float(box.find('y3').text))
            if x0 < 0:
                x0 = 0
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if x3 < 0:
                x3 = 0
            if y0 < 0:
                y0 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0
            if y3 < 0:
                y3 = 0
            for cls_index, cls_name in enumerate(cls_list):
                if cls == cls_name:
                    file.write("{} {} {} {} {} {} {} {} {} {}\n".format(x0, y0, x1, y1, x2, y2, x3, y3, cls, cls_index))
        file.close()
        # print(output)
        print(i)
 
def dir_exists(folder_path):
        # 判断文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")
 
def copy_files_except_last_n(source_folder, destination_folder, n):
    try:
        # 获取源文件夹中的所有文件列表
        files = os.listdir(source_folder)
        
        # 确保文件列表按照文件名排序
        files.sort()
        
        # 选择要复制的文件
        files_to_copy = files[:-n]  # 复制除了最后 n 个文件之外的所有文件
        
        # 创建目标文件夹（如果不存在）
        os.makedirs(destination_folder, exist_ok=True)
        
        # 复制选定的文件到目标文件夹
        for file_name in files_to_copy:
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copy(source_file, destination_file)
        
        print(f"成功复制文件夹 '{source_folder}' 中除了最后 {n} 个文件之外的所有文件到 '{destination_folder}'")
    
    except FileNotFoundError:
        print(f"文件夹 '{source_folder}' 不存在或无法访问。")
    except OSError as e:
        print(f"复制文件时发生OSError: {e}")
 
if __name__ == '__main__':
    # -----**** 第一步：把xml文件统一转换成旋转框的xml文件 ****-----
    
    roxml_path = os.path.join(data_path, 'xml') # 存放roLabelImg标注的原文件的文件夹
    
    # 获取文件夹中所有文件
    files = os.listdir(data_path)

    # 提取jpg和xml文件名（不含扩展名）
    jpg_files = {os.path.splitext(file)[0] for file in files if file.endswith('.jpg')}
    xml_files = {os.path.splitext(file)[0] for file in files if file.endswith('.xml')}

    # 找到没有对应xml文件的jpg文件
    jpg_without_xml = jpg_files - xml_files

    # 删除没有对应xml文件的jpg文件
    for jpg in jpg_without_xml:
        jpg_path = os.path.join(data_path, f'{jpg}.jpg')
        os.remove(jpg_path)
        print(f'Deleted {jpg_path}')
    
    dotaxml_path = os.path.join(data_path, 'DOTA_xml') # 存放转换后DOTA数据集xml格式文件的文件夹
    out_path = os.path.join(data_path, 'DOTA_txt') # 存放转换后DOTA数据集txt格式文件的文件夹
    
    dir_exists(dotaxml_path)
    dir_exists(out_path)
    dir_exists(roxml_path)
    
    # 将所有xml文件移动到xml文件夹中
    print(f'Moved {data_path}.xml to {roxml_path}')
    for xml in xml_files:
        xml_path = os.path.join(data_path, f'{xml}.xml')
        shutil.move(xml_path, roxml_path)
    
    filelist = os.listdir(roxml_path)
 
    print('start editing xml files to DOTA.xml..')
    for file in filelist:
        edit_xml(os.path.join(roxml_path, file), os.path.join(dotaxml_path, file))
 
    # -----**** 第二步：把旋转框xml文件转换成txt格式 ****-----
    print('start convert DOTA.xml to DOTA.txt')
    totxt(dotaxml_path, out_path)
    
    labels_train_path = os.path.join(data_path, 'labels/train_original')
    labels_val_path = os.path.join(data_path, 'labels/val_original')
    
    dir_exists(labels_train_path)
    dir_exists(labels_val_path)

    # 获取out_path文件夹中的所有txt文件
    txt_files = [file for file in os.listdir(out_path) if file.endswith('.txt')]

    # 按文件名排序以确保一致性
    txt_files.sort()

    # 将最后7个文件放到val文件夹中，其余的文件放到train文件夹中
    val_files = txt_files[-7:]
    train_files = txt_files[:-7]

    # 移动文件到对应的文件夹中
    print(f'Moved .txt from {out_path} to {labels_val_path}')
    for file in val_files:
        src_path = os.path.join(out_path, file)
        dst_path = os.path.join(labels_val_path, file)
        shutil.move(src_path, dst_path)
    
    print(f'Moved .txt from {src_path} to {dst_path}')
    for file in train_files:
        src_path = os.path.join(out_path, file)
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
    
    print('start convert Dota to YOLO format...')
    convert_dota_to_yolo_obb(data_path) # DOTA 格式数据集路径，里头包含 images和labels，labels中包含train_original和val_original
    
    shutil.rmtree(dotaxml_path)
    shutil.rmtree(out_path)
    shutil.rmtree(roxml_path)
    shutil.rmtree(data_path + '/labels' + '/train_original')
    shutil.rmtree(data_path + '/labels' + '/val_original')
    
    print('Process completed successfully.')