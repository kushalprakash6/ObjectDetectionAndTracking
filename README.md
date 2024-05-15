# Object Detection and Tracking in Computer Vision using Deep Learning

Project for object detection and tracking using OpenCV in python. This is done as part of my individual project in my masters course
I have used Yolov8 algorithm for object detection and tracking
Intel RealSense D435 camera is used for capturing RGB and depth information.
More information can be found here: [Link to documents](documents)

## The source code is as follows:
1. To view the feed from the sensor [Realsense.py](src/Realsense.py)
2. Training the Yolo model [train.py](src/train.py)
3. Resuming the training in case of interruption [resume_train.py](src/resume_train.py)
4. Capturing RGB video [capture_colourVideo.py](src/capture_colourVideo.py)
5. Capturing depth images [captureDepthImageCyclic.py](src/captureDepthImageCyclic.py)
6. Capture depth video [captureDepthVideo.py](src/captureDepthVideo.py)
7. Testing the prediction [testVideo.py](src/testVideo.py)


## Training data links
1. Depth images used for training [Link](code/data/images/train)
2. Depth images label used for training [Link](code/data/labels/train)


## Output from the training for depth image data
 Link to training result [Link](runs/detect/train3)


 ## RGB images weights for yolo algorithm
  Link weights for yolo in which coco images dataset trained on [Link](yolov8n.pt)
