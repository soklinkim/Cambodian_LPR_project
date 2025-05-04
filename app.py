import streamlit as st
import os
from PIL import Image  # ✅ REQUIRED for displaying uploaded image

# Make sure 'temp' folder exists
os.makedirs("temp", exist_ok=True)

# Your custom functions
from main import process_uploaded_image, crop_characters, read_text_from_crops

st.title("🚗 Cambodian License Plate Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    img_path = os.path.join("temp", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Show the uploaded image
    st.image(Image.open(img_path), caption="Uploaded Image", use_column_width=True)

    # 🔘 Add Detect button
    if st.button("Detect"):
        # Run detection
        enhanced_paths = process_uploaded_image(img_path)

        # Display results
        st.subheader("📌 Detected License Plate Texts:")
        for path in enhanced_paths:
            try:
                char_crops = crop_characters(path)
                predictions = read_text_from_crops(char_crops)

                if len(predictions) >= 2:
                    st.write(f"Plate number detected: `{predictions[0]}`")
                    st.write(f"Plate Region detected: `{predictions[1]}`")
                else:
                    st.write(f"Undetected text Found: This might be because of the image is too small and blur.")
            except Exception as e:
                st.error(f"Error processing {path}: {e}")
