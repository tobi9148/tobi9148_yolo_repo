from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with the correct model path

# Start training
try:
    model.train(data="dice.yaml", epochs=50, batch=12, imgsz=448)
except Exception as e:
    print(f"Error during training: {e}")