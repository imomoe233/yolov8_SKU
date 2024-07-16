#-*-codeing:uft-8-*-
import cv2
import os
import shutil



img_dir = r'D:\code\datasets\candy\shangpin\images\val'
txt_dirs = r"D:\code\datasets\candy\shangpin\labels\yolo_result_txt"
same_name = r"D:\code\datasets\candy\shangpin\labels\yolo_result_xml"
xml_dirs = r"txt2xml"
if  not os.path.exists(xml_dirs):
    os.mkdir(xml_dirs)
else:
    shutil.rmtree(xml_dirs)
    os.mkdir(xml_dirs)

CLASSES = ["0"]
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
            # print('spt',spt)
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

old_name_list = []
for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith((".jpg", ".bmp", ".png",".jpeg")):
            img_path = os.path.join(root, file)
            name = file.rsplit(".", 1)[0]
            if name in old_name_list:
                shutil.move(img_path, same_name)
                continue
            old_name_list.append(name)
            txt_path = os.path.join(txt_dirs, name+".txt")   
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
                    xmin = cx - cw * 0.5
                    xmax = cx + cw * 0.5
                    ymin = cy - ch * 0.5
                    ymax = cy + ch * 0.5
                    angle = num[5]
                    # labels.append([cx,cy,cw,ch,cls])
                    labels.append([cx,cy,cw,ch,cls,angle])
            
            xml_write(img, labels, xml_dirs, file)
            shutil.copy(img_path, xml_dirs)
            
