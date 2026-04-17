# app.py

import cv2
from ultralytics import YOLO

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("runs/detect/train/weights/best.pt")

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not found")
    exit()

# Set camera quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Predict
    results = model.predict(
        source=frame,
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    # Draw predictions
    output = results[0].plot()

    # Show window
    cv2.imshow("Bangladesh Currency Detection", output)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Close
# -----------------------------
cap.release()
cv2.destroyAllWindows()