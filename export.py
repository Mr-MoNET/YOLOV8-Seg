from ultralytics import YOLO

# Load a model
PremodelPath = '/home/gao/Desktop/yolov8/yolov8s-seg.pt'
#modelPath = '/home/gao/Desktop/yolov8/runs/train/seg-0802/weights/best.pt'
modelPath = '/home/gao/Desktop/yolov8/yolov8s-seg.pt'

model = YOLO(PremodelPath)  # load an official model
model = YOLO(modelPath)  # load a custom trained model

# Export the model
success = model.export(format="onnx", imgsz=416)

# Check whether the export was successful  
assert success
