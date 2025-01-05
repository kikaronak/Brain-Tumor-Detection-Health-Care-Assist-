import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('brain_tumor_model.h5')

# Class names for prediction
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']  # Replace with your actual class names

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((124, 124))  # Resize to the model's input shape
    image = img_to_array(image)      # Convert to NumPy array
    image = image / 255.0           # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Streamlit UI
st.title("Brain Tumor Prediction App")
st.write("Upload an MRI image to predict the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    st.write("Processing the image...")

    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    # Display the prediction
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {np.max(prediction) * 100:.2f}%")
