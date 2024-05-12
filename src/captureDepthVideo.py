import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
out = cv2.VideoWriter('depth_video.mp4', fourcc, 30.0, (640, 480))

duration = 15  # in seconds
start_time = cv2.getTickCount()

while True:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    # Convert depth_image to 8-bit for histogram equalization
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=1)
    # Equalize the histogram of the depth image
    equalized_depth = cv2.equalizeHist(depth_image_8bit)
    # Apply colormap
    depth_cm = cv2.applyColorMap(equalized_depth, cv2.COLORMAP_JET)

    # Write frame to the video
    out.write(depth_cm)

    cv2.imshow('depth', depth_cm)

    current_time = cv2.getTickCount()
    # Break the loop if duration is exceeded
    if (current_time - start_time) / cv2.getTickFrequency() > duration:
        break

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
out.release()
cv2.destroyAllWindows()
