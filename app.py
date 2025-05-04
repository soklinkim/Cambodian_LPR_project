import streamlit as st
import os
from PIL import Image

# Ensure the 'temp' directory exists
os.makedirs("temp", exist_ok=True)

# Import your custom functions
from main import process_uploaded_image, crop_characters, read_text_from_crops

# ---------------------- Page Config & Styling ----------------------
st.set_page_config(page_title="Cambodian License Plate Recognition", layout="centered")

st.markdown("""
    <style>
    .title {
        font-size:36px;
        text-align:center;
        color:#2C3E50;
        font-weight:bold;
    }
    .subtitle {
        font-size:20px;
        color:#34495E;
        margin-top:10px;
    }
    .footer {
        font-size:14px;
        color:#95A5A6;
        margin-top:50px;
        text-align:center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚗 Cambodian License Plate Recognition</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Upload a vehicle image to read its license plate</div>', unsafe_allow_html=True)

# ---------------------- Upload Image ----------------------
uploaded_file = st.file_uploader("📤 Upload a vehicle image (JPG/PNG) to read its license plate", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save image to temporary path
    img_path = os.path.join("temp", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(Image.open(img_path), caption="✅ Uploaded Image", use_column_width=True)

    # Detect button
    if st.button("🔍 Detect License Plate"):
        try:
            with st.spinner("Processing... please wait."):
                enhanced_paths = process_uploaded_image(img_path)

            st.subheader("📌 Detection Results:")
            for path in enhanced_paths:
                try:
                    char_crops = crop_characters(path)
                    predictions = read_text_from_crops(char_crops)

                    if len(predictions) >= 2:
                        st.success(f"**Plate Number**: `{predictions[0]}`")
                        st.info(f"**Plate Region**: `{predictions[1]}`")
                    else:
                        st.warning("⚠️ Partial text detected. The image may be blurry or too small.")
                except Exception as inner_e:
                    st.error(f"🚫 Error processing image: {inner_e}")
        except Exception as outer_e:
            st.error(f"Unexpected error during processing: {outer_e}")


st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center; font-size: 14px; color: #7F8C8D;">
        <strong>Developed by:</strong> AI Girls 👩‍💻<br>
        <strong>Team Members:</strong> KIM Soklin, CHEA Sreymom, SENG Mouyheang, NIN Phallei, CHHEA Muoyheang<br>
        <strong>Class:</strong> AI Class (ITM 360)<br>
        <strong>Professor:</strong> Prof. PIN Kuntha<br><br>
        &copy; 2025 American University of Phnom Penh
    </div>
""", unsafe_allow_html=True)