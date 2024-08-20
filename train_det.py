from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    data_yaml = '/home/gao/Desktop/yolov8/task/detect/custom.yaml'
    pre_model = '/home/gao/Desktop/yolov8/model/yolov8s.pt'

    model = YOLO(pre_model, task='detect')

    model.train(data=data_yaml, 
                epochs=300, 
                imgsz=416, 
                batch=32, 
                device=0,
                project='runs/train',
                name='det-0820',
                cos_lr=True, 
                amp=False,
                patience=100)
    