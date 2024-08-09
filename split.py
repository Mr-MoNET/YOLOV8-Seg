# split.py 用于数据集划分为训练集 验证集 测试集，可以自行设置划分比例

import os
import shutil
import random
from tqdm import tqdm

# 数据集文件夹路径
dataset_dir = '/home/gao/Desktop/twoclass'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 输出文件夹路径
split_dir = '/home/gao/Desktop/yolov8/split'
output_dirs = {
    'train': os.path.join(split_dir, 'train'),
    'val': os.path.join(split_dir, 'val'),
    'test': os.path.join(split_dir, 'test')
}

# 创建输出文件夹
for split in output_dirs:
    os.makedirs(os.path.join(output_dirs[split], 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dirs[split], 'labels'), exist_ok=True)

# 获取所有图像和标签文件名
image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
label_files = sorted([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])

# 确保图像文件和标签文件一一对应
assert len(image_files) == len(label_files)

# 打乱文件顺序
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files[:], label_files[:] = zip(*combined)

# 按照8:1:1的比例划分数据集
total_files = len(image_files)
train_end = int(total_files * 0.8)
val_end = train_end + int(total_files * 0.1)

# 分配文件到相应的文件夹
splits = {
    'train': (0, train_end),
    'val': (train_end, val_end),
    'test': (val_end, total_files)
}

# 使用 tqdm 显示进度条
for split, (start, end) in splits.items():
    for i in tqdm(range(start, end), desc=f"分割 {split} 集"):
        shutil.copy(os.path.join(images_dir, image_files[i]), os.path.join(output_dirs[split], 'images', image_files[i]))
        shutil.copy(os.path.join(labels_dir, label_files[i]), os.path.join(output_dirs[split], 'labels', label_files[i]))

print("数据集划分完成！")
