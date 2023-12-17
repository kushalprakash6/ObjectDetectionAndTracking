import cv2

# Capture the webcam video stream
cap = cv2.VideoCapture(0)

# Create a window to display the webcam feed
cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        break

    # Display the frame in the window
    cv2.imshow('Webcam', frame)

    # Check for keyboard exit events
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
