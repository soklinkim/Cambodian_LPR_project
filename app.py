import os
import numpy as np
from PIL import Image
import streamlit as st
from improved_main import detect_license_plates, enhance_plates, crop_characters, read_text_from_crops
# Configuration
MODEL1_PATH = "models/Cambodian_plate_detection.pt"
MODEL2_PATH = "models/best.pt"
CROP_DIR = "output/plate_crops"
ENHANCED_DIR = "output/plate_crops_enhanced"

# Make output directories
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

# Streamlit app code
st.title('Cambodian License Plate Recognition')
st.write("Upload an image of a vehicle for license plate detection, enhancement, and text recognition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect License Plates'):
        st.write("Detecting license plates...")
        image, results, cropped_images = detect_license_plates(image, MODEL1_PATH)
        
        # Show cropped images
        for idx, cropped_img in enumerate(cropped_images):
            st.image(cropped_img, caption=f'Cropped Plate {idx + 1}')
        
        # Enhance the plates
        st.write("Enhancing plates...")
        enhanced_paths = enhance_plates(image, results)
        for path in enhanced_paths:
            st.image(path, caption=f'Enhanced Plate {os.path.basename(path)}')

        # Crop characters and read text
        st.write("Reading text from enhanced crops...")
        for path in enhanced_paths:
            char_crops = crop_characters(path, MODEL2_PATH)
            predictions = read_text_from_crops(char_crops)
            st.write(f"Predictions for {path}: {predictions}")
