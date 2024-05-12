# import cv2

# def capture_video(duration, resolution=(640, 480), fps=30):
#     # Open the webcam
#     cap = cv2.VideoCapture(0)
    
#     # Set resolution and frame rate
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
#     cap.set(cv2.CAP_PROP_FPS, fps)
    
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
#     out = cv2.VideoWriter('captured_video1.mp4', fourcc, fps, resolution)
    
#     # Capture video for the specified duration
#     start_time = cv2.getTickCount() / cv2.getTickFrequency()
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             out.write(frame)
#         current_time = cv2.getTickCount() / cv2.getTickFrequency()
#         if (current_time - start_time) >= duration:
#             break
    
#     # Release everything when done
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Capture video for 5 seconds
# capture_video(5)

import cv2

def capture_video(duration, resolution=(640, 480), fps=30):
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Set resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter('captured_video10.mp4', fourcc, fps, resolution)
    
    # Capture video for the specified duration
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Preview', frame)  # Show preview
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording
                break
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if (current_time - start_time) >= duration:
            break
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Capture video for 5 seconds
capture_video(5)
