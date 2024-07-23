import os
import shutil
import subprocess


# Define paths
data_path = r'D:\code\datasets\drink_hobby_seasoning\bantai\huojia'

#jdclj
label_list = 'shtllj,shpjykj,ljbllj,ljyklj,ljhjzj,shpqt' 

# 获取文件夹中所有文件
files = os.listdir(data_path)

# 提取jpg和xml文件名（不含扩展名）
jpg_files = {os.path.splitext(file)[0] for file in files if file.endswith('.jpg')}
json_files = {os.path.splitext(file)[0] for file in files if file.endswith('.json')}

# 找到没有对应xml文件的jpg文件
jpg_without_xml = jpg_files - json_files

# 删除没有对应xml文件的jpg文件
for jpg in jpg_without_xml:
    jpg_path = os.path.join(data_path, f'{jpg}.jpg')
    os.remove(jpg_path)
    print(f'Deleted {jpg_path}')



def dir_exists(folder_path):
        # 判断文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")

def clean_directory(path, folders_to_keep):
    # List all items in the specified directory
    items = os.listdir(path)

    # Loop through the items and delete those that are not in the folders_to_keep list
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item not in folders_to_keep:
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)



# 构建命令
command = [
    'labelme2yolo',  # 这里假设你的脚本名是 `convert_labelme_to_yolo.py`
    '--val_size', '0.02',
    '--json_dir', data_path,
    '--label_list', label_list
]

yolo_dataset_path = 'YOLODataset'
images_yolo_path = os.path.join(data_path, yolo_dataset_path, 'images')
labels_yolo_path = os.path.join(data_path, yolo_dataset_path, 'labels')

#dir_exists(images_yolo_path)
#dir_exists(labels_yolo_path)

images_path = os.path.join(data_path, 'images')
labels_path = os.path.join(data_path, 'labels')

#dir_exists(images_path)
#dir_exists(labels_path)

# 运行命令
subprocess.run(command)

shutil.copytree(images_yolo_path, images_path)
shutil.copytree(labels_yolo_path, labels_path)

# Delete YOLODataset directory
shutil.rmtree(os.path.join(data_path, yolo_dataset_path))

# Specify the folders to keep
folders_to_keep = ['images', 'labels']

# Clean the directory
# clean_directory(data_path, folders_to_keep)

print("Process completed successfully.")
