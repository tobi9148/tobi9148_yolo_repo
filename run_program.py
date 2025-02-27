# filepath: /c:/Users/tobia/Desktop/Skole/Digitek/Digitek24_25/machine_learning/Yolo/classify_webcam.py
import cv2
import numpy as np
from ultralytics import YOLO
import json

# Load YOLO model
model = YOLO("runs/detect/train22/weights/best.pt")  # Replace with the correct model path

# Load class names from dice.yaml
class_names = ['Dice1', 'Dice2', 'Dice3', "Dice4", "Dice5", "Dice6"]

# Define colors for each class
colors = {
    'Dice1': (255, 0, 0),  # Blue
    'Dice2': (0, 165, 255),  # Green
    'Dice3': (0, 0, 255),  # Red
    'Dice4': (255, 255, 0),  # Cyan
    'Dice5': (255, 0, 255),  # Magenta
    'Dice6': (0, 255, 255)   # Yellow
}

# Function to draw bounding boxes on the frame
# Function to draw bounding boxes on the frame and display the total sum of dice
def draw_boxes(frame, results):
    bounding_boxes = []
    total_sum = 0
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for i in range(len(boxes)):
            x, y, x2, y2 = boxes[i]
            confidence = confidences[i]
            if confidence >= 0.66:  # Only draw boxes if confidence is over 75%
                if class_ids[i] < len(class_names):
                    label = str(class_names[class_ids[i]])
                    dice_value = int(label[-1])  # Extract the dice value from the label
                    total_sum += dice_value
                else:
                    label = "Unknown"
                color = colors.get(label, (0, 255, 0))  # Default to green if label not found
                cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                bounding_boxes.append({
                    "label": label,
                    "confidence": float(confidence),
                    "box": [int(x), int(y), int(x2), int(y2)]
                })

    # Display the total sum of dice in the top center of the frame
    frame_height, frame_width = frame.shape[:2]
    cv2.putText(frame, f"Total Sum: {total_sum}", (frame_width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame, bounding_boxes

def classify_webcam():
    # Try different camera indices
    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Using camera index {camera_index}")
            break
    else:
        raise RuntimeError("No available camera found")

    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)

    bounding_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Perform object detection
        results = model(frame)

        # Debugging: Print results
        print(f"Results: {results}")

        # Draw bounding boxes on the frame
        frame, boxes = draw_boxes(frame, results)
        bounding_boxes.extend(boxes)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return bounding_boxes

if __name__ == "__main__":
    bounding_boxes = classify_webcam()
    print(json.dumps(bounding_boxes))

# Try different camera indices
for camera_index in range(5):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Using camera index {camera_index}")
        break
else:
    raise RuntimeError("No available camera found")

cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Perform object detection
    results = model(frame)

    # Debugging: Print results
    print(f"Results: {results}")

    # Draw bounding boxes on the frame
    frame = draw_boxes(frame, results)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()