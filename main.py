import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from ultralytics import YOLO

# Configuration
INPUT_IMAGE_PATH = "data/car2.jpg"
MODEL1_PATH = "models/Cambodian_plate_detection.pt"
MODEL2_PATH = "models/best.pt"
CROP_DIR = "output/plate_crops"
ENHANCED_DIR = "output/plate_crops_enhanced"

# Make output directories
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

# Load models once
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Using device: {device}")

plate_model = YOLO(MODEL1_PATH)  # Plate detection
char_model = YOLO(MODEL2_PATH)   # Character detection

ocr_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True)
ocr_model = ocr_model.to(device).eval()

# Transform for OCR input
ocr_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

def detect_license_plate(image_array, model_path):
    model = YOLO(model_path)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)[0]

    cropped_images = []
    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image_array[y1:y2, x1:x2]
        filename = os.path.join(CROP_DIR, f"plate_{idx + 1}.jpg")
        cv2.imwrite(filename, cropped)
        cropped_images.append(cropped)  # Add image array to return

    return image_array, results, cropped_images


def enhance_plates(image, results):
    enhanced_paths = []
    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        sharpened = img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        img_np = np.array(sharpened)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(img_gray)
        final_img = Image.fromarray(equalized).convert('RGB').resize((640, 640), Image.Resampling.LANCZOS)
        path = os.path.join(ENHANCED_DIR, f"plate_{idx+1}_enhanced.jpg")
        final_img.save(path)
        enhanced_paths.append(path)
    return enhanced_paths

def crop_characters(image_path):
    results = char_model(image_path)
    orig_img = cv2.imread(image_path)
    boxes = results[0].boxes
    if len(boxes) < 2:
        return []
    top2 = sorted(boxes, key=lambda b: b.conf.item(), reverse=True)[:2]
    crops = []
    for box in top2:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().astype(int))
        cropped = orig_img[y1:y2, x1:x2]
        h = 64
        cropped = cv2.resize(cropped, (int(cropped.shape[1] * h / cropped.shape[0]), h))
        crops.append(cropped)
    return crops

def read_text_from_crops(crops):
    predictions = []
    for crop in crops:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = ocr_transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = ocr_model(image_input).softmax(-1)
            decoded = ocr_model.tokenizer.decode(pred)[0][0]
        predictions.append(decoded)
    return predictions

def main():
    # Load the image
    image_array = cv2.imread(INPUT_IMAGE_PATH)
    
    # Pass the image and the model path to the function
    image, results, cropped_images = detect_license_plate(image_array, MODEL1_PATH)
    
    # Enhance the plates (optional)
    enhanced_paths = enhance_plates(image, results)
    
    # Process character recognition for each enhanced plate
    for path in enhanced_paths:
        char_crops = crop_characters(path)
        predictions = read_text_from_crops(char_crops)
        if len(predictions) >= 2:
            print(f"Plate number detected: '{predictions[0]}'")
            print(f"Plate Region detected: '{predictions[1]}'")
        else:
            print(f"Detected text: {predictions}")

def process_uploaded_image(image_path):
    image_array = cv2.imread(image_path)
    image, results, cropped_images = detect_license_plate(image_array, MODEL1_PATH)
    enhanced_paths = enhance_plates(image, results)
    return enhanced_paths

if __name__ == "__main__":
    main()
