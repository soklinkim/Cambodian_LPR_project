import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms
from ultralytics import YOLO

# Configuration
INPUT_IMAGE_PATH = "data/car1.jpg"
MODEL1_PATH = "models/Cambodian_plate_detection.pt"
MODEL2_PATH = "models/best.pt"
CROP_DIR = "output/plate_crops"
ENHANCED_DIR = "output/plate_crops_enhanced"

# Make output directories
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

def detect_license_plates(image_path, model_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)[0]
    
    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        filename = os.path.join(CROP_DIR, f"plate_{idx + 1}.jpg")
        cv2.imwrite(filename, cropped)
    return image, results

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

def crop_characters(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
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
    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    predictions = []
    for crop in crops:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = transform(crop_pil).unsqueeze(0).to(model.device)
        with torch.no_grad():
            pred = model(image_input).softmax(-1)
            decoded = model.tokenizer.decode(pred)[0][0]
        predictions.append(decoded)
    return predictions

def main():
    image, results = detect_license_plates(INPUT_IMAGE_PATH, MODEL1_PATH)
    enhanced_paths = enhance_plates(image, results)
    for path in enhanced_paths:
        char_crops = crop_characters(path, MODEL2_PATH)
        predictions = read_text_from_crops(char_crops)
        print(f"✅ Predictions for {path}: {predictions}")

if __name__ == "__main__":
    main()
