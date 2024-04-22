import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('facial_recognition_model.keras')

# Function to preprocess input frame
def preprocess_frame(frame):
    # Resize frame to match input dimensions expected by the model
    # You may need to apply other preprocessing steps here (e.g., normalization)
    processed_frame = cv2.resize(frame, (100, 100))  # Adjust dimensions as per your model's input shape
    return processed_frame

# Function to perform prediction on a frame
def predict_frame(frame):
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    # Reshape frame to match the input shape expected by the model
    input_frame = np.expand_dims(processed_frame, axis=0)
    # Perform prediction
    prediction = model.predict(input_frame)
    # Example: Assuming binary classification (0 or 1)
    if prediction < 0.4:
        return "Rahul"
    elif prediction > 0.6:
        return "Melwin"
    else:
        return "No Match"

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Draw rectangle around scanning area
    top_left = (100, 100)  # Top left corner of the rectangle
    bottom_right = (300, 300)  # Bottom right corner of the rectangle
    color = (0, 255, 0)  # Color of the rectangle (green)
    thickness = 2  # Thickness of the rectangle border
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    # Perform prediction on the frame
    result = predict_frame(frame)
    # Display the frame and prediction result
    cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
