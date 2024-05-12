# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('/Users/kushalprakash/yolov8/runs/detect/train2/weights/best.pt')  # load a custom model

# # Predict with the model
# results = model('/Users/kushalprakash/Desktop/untitled folder/new_image_2999.jpg')  # predict on an image
# print(results)

# # View results
# for r in results:
#     print(r.boxes)  # print the Boxes object containing the detection bounding boxes


# from ultralytics import YOLO

# # Load a pretrained YOLOv8n model
# model = YOLO('yolov8n.pt')
# model = YOLO('/Users/kushalprakash/yolov8/runs/detect/train2/weights/best.pt')  # load a custom model

# # Run inference on 'bus.jpg' with arguments
# boxes, classes, scores = model.predict('/Users/kushalprakash/Desktop/untitled folder/new_image_1351.jpg', save=True, imgsz=320, conf=0.5)

# if boxes is not None:
#     draw(image, boxes, scores, classes, all_classes)

import torch
import cv2
from torchvision import transforms
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Load YOLOv5 model
model = attempt_load('/Users/kushalprakash/yolov8/runs/detect/train2/weights/best.pt', map_location=torch.device('cpu'))  # Change the path according to your model

# Set the model to evaluation mode
model.eval()

# Define a function to perform object detection and display the results
def detect_objects(image_path):
    img0 = cv2.imread(image_path)  # Load image

    # Resize image to a square shape with letterboxing
    img = letterbox(img0, new_shape=model.img_size)[0]

    # Convert image to Torch tensor
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0

    # Add batch dimension
    img = img.unsqueeze(0)

    # Detect objects
    with torch.no_grad():
        prediction = model(img)

    # Apply non-maximum suppression
    prediction = non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.5)

    # Display results
    for i, det in enumerate(prediction):
        if det is not None and len(det):
            # Rescale bounding boxes to original image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # Draw bounding boxes
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                # Display class label and confidence
                label = f'{model.names[int(cls)]}: {conf:.2f}'
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Detection Output', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function with an example image
detect_objects('/Users/kushalprakash/Desktop/untitled folder/new_image_1351.jpg')  # Replace 'test_image.jpg' with your image path
