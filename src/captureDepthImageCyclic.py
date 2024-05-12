import os
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Create a folder to store images if it doesn't exist
folder_name = "depth_images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

count = 0
start_time = time.time()
interval = 0.5  # in seconds
max_images = 100

while count < max_images:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    # Convert depth_image to 8-bit for histogram equalization
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=1)
    # Equalize the histogram of the depth image
    equalized_depth = cv2.equalizeHist(depth_image_8bit)
    # Apply colormap
    depth_cm = cv2.applyColorMap(equalized_depth, cv2.COLORMAP_JET)

    # Show depth image
    cv2.imshow('depth', depth_cm)

    # Save depth image every 0.5 seconds
    current_time = time.time()
    if current_time - start_time >= interval:
        filename = os.path.join(folder_name, f'depth_image_{count}.jpeg')
        cv2.imwrite(filename, depth_cm)
        print(f"Saved {filename}")
        count += 1
        start_time = current_time

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()
