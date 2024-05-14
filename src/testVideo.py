import os
import torch
from ultralytics import YOLO
import cv2

'''
    Select the video pair based on what is being tested or run
    One set of video is in normal light conditions and the other is in dark condotions
'''


# video_path_depth = '/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/depth_video_bright.mp4'
# video_path_rgb = '/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/rgb_video_bright.mp4'
# output_file_path_rgb = "/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/detected_objects_rgb_bright.txt"
# output_file_path_depth = "/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/detected_objects_depth_bright.txt"
# consolidated_file = '/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/consolidated_detected_objects_bright.txt'


video_path_depth = '/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/depth_video_dark.mp4'
video_path_rgb = '/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/rgb_video_dark.mp4'
output_file_path_rgb = "/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/detected_objects_rgb_dark.txt"
output_file_path_depth = "/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/detected_objects_depth_dark.txt"
consolidated_file = '/Users/kushalprakash/ObjectDetectionAndTracking/sampleVideos/consolidated_detected_objects_dark.txt'



# RGB video is parsed and marked here

video_path_out = '{}_out.mp4'.format(video_path_rgb)

cap = cv2.VideoCapture(video_path_rgb)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

model_path = '/Users/kushalprakash/Downloads/yolov8n.pt'

# Load a model
model = YOLO(model_path)  # load a custom model


threshold = 0.6
# Open a text file for writing
output_file = open(output_file_path_rgb, "w")

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        class_name = results.names[int(class_id)].upper()
        object_info = f"Class: {class_name}, Score: {score}, Bounding Box: ({x1}, {y1}), ({x2}, {y2})"
        output_file.write(object_info + "\n")

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()


# depth video is parsed and marked here

video_path_out = '{}_out.mp4'.format(video_path_depth)

cap = cv2.VideoCapture(video_path_depth)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

model_path = '/Users/kushalprakash/yolov8_trained/yolo/runs/detect/train3/weights/best.pt'

# Load a model
model = YOLO(model_path)  # load a custom model


threshold = 0.6
# Open a text file for writing

output_file = open(output_file_path_depth, "w")

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        class_name = results.names[int(class_id)].upper()
        object_info = f"Class: {class_name}, Score: {score}, Bounding Box: ({x1}, {y1}), ({x2}, {y2})"
        output_file.write(object_info + "\n")

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()



def extract_score(line):
    """Extract the score from a line."""
    score_index = line.find("Score: ") + len("Score: ")
    score_str = line[score_index:].split(",")[0].strip()
    return float(score_str)

def compare_scores(file1, file2, output_file):
    """Compare scores from two files and create a new file."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as output:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    max_total_lines = max(len(lines1), len(lines2))

    with open(output_file, 'w') as output:
        for i in range(max_total_lines):
            if i < len(lines1) and lines1[i].strip():
                score1 = extract_score(lines1[i])
            else:
                score1 = float('-inf')
                
            if i < len(lines2) and lines2[i].strip():
                score2 = extract_score(lines2[i])
            else:
                score2 = float('-inf')
        
            if score1>= 0.9 and score2 >= 0.9:
                print(f"{i+1}: both sensor has high confidence score. {lines1[i]}")
                print("")
                output.write(f"{lines1[i].strip()}, sensor: both\n")
            elif score1 > score2:
                print(f"{i+1}: rgb sensor has high confidence score. {lines1[i]}")
                print("")
                output.write(f"{lines1[i].strip()}, sensor: rgb\n")
            elif score1 < score2:
                print(f"{i+1}: depth sensor has high confidence score. {lines2[i]}")
                print("")
                output.write(f"{lines2[i].strip()}, sensor: depth\n")
            elif score1<= 0.2 and score2 <= 0.2:
                print(f"{i+1}: both sensor has low confidence score. {lines1[i]}")
                print("")
                output.write(f"{lines1[i].strip()}, sensor: both\n")
            else:
                print(f"Line {i+1}: Scores in both files are equal: {score1} = {score2}")
                print("")
                output.write(f"{lines1[i].strip()}, sensor: both\n")
                # If scores are equal, you can choose to write to output file or not
                pass


# text files paths
file1 = output_file_path_rgb
file2 = output_file_path_depth

compare_scores(file1, file2, consolidated_file)
