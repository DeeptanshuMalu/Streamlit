import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

st.title("Crack Detection and Measurement")

# Function to load images and their corresponding segmentation masks
def load_images_and_masks(image_path, mask_path):
    images = []
    masks = []
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg"):
            # Load the image
            image = cv2.imread(os.path.join(image_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 128))  # Resize images to a common size
            images.append(image)

            # Load the corresponding mask image
            mask_filename = filename.split('.')[0] + '.png'  # Assuming mask images have _mask.png extension
            mask_img = cv2.imread(os.path.join(mask_path, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (128, 128))  # Resize mask images
            masks.append(mask_img)

    return np.array(images), np.array(masks)

uploaded_image=st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_image is not None:

    img = Image.open(uploaded_image)

    # Load the trained segmentation model
    model = load_model('crack_segmentation_model.h5')  # Load your trained model file

    # Load your own image
    image = np.array(img)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_shape = image_gray.shape[:2]  # Original image dimensions

    # Preprocess the image
    image_resized = cv2.resize(image_gray, (128, 128))  # Resize image to match the input size used during training
    image_resized = image_resized.astype('float32') / 255.0
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

    # Make predictions on your image
    predictions = model.predict(image_resized)

    # Threshold the predictions to obtain binary mask (crack or no crack)
    threshold = 0.5  # You can adjust this threshold as needed
    binary_mask = (predictions > threshold).astype(np.uint8)
    resized_binary_mask = cv2.resize(binary_mask[0], (original_shape[1], original_shape[0]))  # Resize mask to original image dimensions

    col1, col2=st.columns(2)

    col1.image(img, "Original")
    col2.image(resized_binary_mask*255, "Predicted Mask")

    # Find contours in the resized binary mask
    contours, _ = cv2.findContours(resized_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate measurements of contours (length, width, depth)
    i=0
    for contour in contours:
        # Calculate the area of the contour (crack area)
        area = cv2.contourArea(contour)
        st.write(f"{i+1}.")
        
        # Calculate the length of the contour (perimeter)
        length = cv2.arcLength(contour, closed=True)
        
        # Calculate the minimum enclosing circle to get width
        (x, y), radius = cv2.minEnclosingCircle(contour)
        width = 2 * radius  # Diameter of the circle
        
        # Display measurements
        st.markdown(f"- Crack Area: {area:.2f} square units")
        st.markdown(f"- Crack Length: {length:.2f} units")
        st.markdown(f"- Crack Width: {width:.2f} units")
        i+=1