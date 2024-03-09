import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained CNN model for hand gesture recognition
model = load_model('FinalModel_TransferLearning_VGG19.h5')  # Replace with the path to your model

# Dictionary to map gesture labels to menu items or commands
gesture_menu_mapping = {
     0: "Pizza",
    1: "Burger",
    2: "Pasta",
    3: "Salad",
    4: "Sandwich",
    5: "Sushi",
    6: "Taco",
    7: "Hot Dog",
    8: "Ice Cream",
    9: "French Fries",
    10: "Chicken Wings",
    11: "Steak",
    12: "Soup",
    13: "Dumplings",
    14: "Noodles",
    15: "Fried Rice",
    16: "Fish and Chips",
    17: "Fried Chicken",
    18: "Shrimp",
    19: "Nachos",
    20: "Cheese",
    21: "Vegetables",
    22: "Fruit",
    23: "Cake",
    24: "Coffee",
    25: "Tea"
}

# Function to preprocess the image and perform prediction
def predict_gesture(image):
    # Preprocess the image (resizing, normalization, etc.)
    image = cv2.resize(image, (50, 50))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform prediction using the loaded model
    prediction = model.predict(image)
    gesture_label = np.argmax(prediction)
    return gesture_label

def main():
    st.title('Hand Gesture Recognition for Restaurant Ordering')

    # Create a webcam object
    cap = cv2.VideoCapture(0)

    # Add a button to process a single frame
    if st.button("Process Frame"):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gesture_label = predict_gesture(frame)
        gesture = gesture_menu_mapping.get(gesture_label, "Unknown Gesture")

        # Display the recognized gesture on the Streamlit app
        st.write(f"Recognized Gesture: {gesture}")

        # Display the webcam feed
        st.image(frame, channels='BGR', use_column_width=True)

    cap.release()

if __name__ == '__main__':
    main()
