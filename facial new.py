import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to load and resize images from a folder
def load_and_resize_images(folder, target_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            images.append(img_resized)
            labels.append(0 if folder == "Rahul" else 1)  # Encode labels as integers
    return images, labels

# Load and resize images from the folders
images = []
labels = []
for folder_name in ["Rahul", "Melwin"]:
    folder_images, folder_labels = load_and_resize_images(folder_name)
    images.extend(folder_images)
    labels.extend(folder_labels)

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model
loss, accuracy = model.evaluate(val_images, val_labels)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Save the model
model.save('facial_recognition_model.keras')

# Load the model for predictions
saved_model = load_model('facial_recognition_model.keras')

# Define a function to predict the class of an image
def predict_image_class(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (100, 100))
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = saved_model.predict(img_resized)
    if prediction < 0.5:
        return "Rahul"
    else:
        return "Melwin"

# Test the model on images in the Rahul folder
print("Predictions for images in Rahul folder:")
for filename in os.listdir("Rahul"):
    img_path = os.path.join("Rahul", filename)
    predicted_class = predict_image_class(img_path)
    print(f"{filename}: Predicted as {predicted_class}")

# Test the model on images in the Melwin folder
print("\nPredictions for images in Melwin folder:")
for filename in os.listdir("Melwin"):
    img_path = os.path.join("Melwin", filename)
    predicted_class = predict_image_class(img_path)
    print(f"{filename}: Predicted as {predicted_class}")
