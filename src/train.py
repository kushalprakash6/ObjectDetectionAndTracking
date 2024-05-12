from ultralytics import YOLO

# Load a model
#yolov8n model is used for parameter setting
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model to train
model.train(data="config.yaml", epochs=100)  # train the model