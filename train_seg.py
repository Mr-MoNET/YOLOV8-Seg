from ultralytics import YOLO
 
if __name__ == '__main__':
    modelpath = '/home/gao/Desktop/yolov8/yolov8s-seg.pt'
    yamlpath = '/home/gao/Desktop/yolov8/split/twoclass.yaml'
 
    model = YOLO(modelpath)
    model.train(epochs=600,
                data=yamlpath,
                imgsz=416,
                batch=32,
                device=0,
                project='runs/train',
                name='seg-0802',
                cos_lr=True,
                amp=False,
                patience=100)