import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = load_model("custom_fire_detection_model.h5")
threshold = 0.4

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess frame
    img = cv2.resize(frame, (64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    label = "🔥 Fire detected!" if prediction[0] > threshold else "✅ Safe"
    color = (0, 0, 255) if "Fire" in label else (0, 255, 0)

    # Show label on frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Live Fire Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
