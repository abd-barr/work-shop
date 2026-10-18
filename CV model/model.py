"""YOLOv8 Webcam Object Detection"""

import cv2
from ultralytics import YOLO

# 1. Load the pretrained YOLOv8n nano model
model = YOLO("best.pt")

# 2. Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam.")

print("Press 'q' in the video window to quit.")

try:
    while True:
        # 3. Read one image (frame) from the webcam
        success, frame = cap.read()

        if not success:
            print("Could not read a frame.")
            break

        # 4. Run object detection
        results = model(frame)

        # 5. Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # 6. Display the detected objects
        cv2.imshow("YOLOv8 Webcam Object Detection", annotated_frame)

        # 7. Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()