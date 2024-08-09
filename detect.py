import os
from ultralytics import YOLO

# 加载模型，并指定使用GPU
modelPath = '/home/gao/Desktop/yolov8/runs/train/seg-0802/weights/best.pt'
model = YOLO(modelPath)
model.to(0)

# 定义单张图片的推理函数
def predict_image(image_path, output_folder):
    try:
        # 推理并保存结果
        model.predict(source=image_path, imgsz=416, save=True, project=output_folder, name='', exist_ok=True)
        print(f"Processed {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 获取文件夹中的所有图片文件
def get_image_files(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(supported_formats)]

# 处理文件夹中的所有图片
def process_folder(input_folder, output_folder):
    # 创建输出文件夹，如果不存在
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = get_image_files(input_folder)
    
    for image_path in image_files:
        predict_image(image_path, output_folder)

# 指定输入图片文件夹路径和输出文件夹路径
input_folder = '/home/gao/Desktop/yolov8/split/test/images'
output_folder = '/home/gao/Desktop/yolov8/runs/detect'

# 开始处理
process_folder(input_folder, output_folder)
