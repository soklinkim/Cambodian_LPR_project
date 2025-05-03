import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from ultralytics import YOLO

# Configuration
MODEL1_PATH = "models/Cambodian_plate_detection.pt"
MODEL2_PATH = "models/best.pt"
CROP_DIR = "output/plate_crops"
ENHANCED_DIR = "output/plate_crops_enhanced"

# Make output directories
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

# Load models once
plate_model = YOLO(MODEL1_PATH)
char_model = YOLO(MODEL2_PATH)
parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).to('cuda' if torch.cuda.is_available() else 'cpu').eval()

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

def enhance_plate(cropped):
    img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    sharpened = img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    img_np = np.array(sharpened)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(img_gray)
    final_img = Image.fromarray(equalized).convert('RGB').resize((640, 640), Image.Resampling.LANCZOS)
    return final_img

def crop_characters(image_pil):
    image_path = "temp.jpg"
    image_pil.save(image_path)
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

def read_text(crops):
    predictions = []
    for crop in crops:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = transform(crop_pil).unsqueeze(0).to(parseq_model.device)
        with torch.no_grad():
            pred = parseq_model(image_input).softmax(-1)
            decoded = parseq_model.tokenizer.decode(pred)[0][0]
        predictions.append(decoded)
    return predictions

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = plate_model(image_rgb)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cropped = frame[y1:y2, x1:x2]
            enhanced = enhance_plate(cropped)
            char_crops = crop_characters(enhanced)
            predictions = read_text(char_crops)
            plate_text = " ".join(predictions)

            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Real-Time License Plate Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
