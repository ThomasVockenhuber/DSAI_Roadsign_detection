import torch
from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model_path = "./20epochs/weights/best.pt"  # Replace with the actual path to your model
model = YOLO(model_path)

# Open a video capture (use 0 for webcam or replace with video file path)
capture = cv2.VideoCapture(0)  # Replace 0 with 'video.mp4' if using a video file

def swaplabel(label):
    # Manually swap incorrect labels
    if label == "speed_limit_30_en":
        label = "STOP"
    elif label == "STOP":
        label = "speed_limit_30_en"

    return label

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    # Run inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label
            label = swaplabel(label)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Road Sign Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
