from pathlib import Path
import argparse
import json

import cv2
import pytesseract
from ultralytics import YOLO

from crop_plates import crop_with_padding
from clean_text import clean_plate_text

ROOT = Path(__file__).resolve().parent


class PlatePipeline:
    def __init__(self, confidence=0.25):
        self.model = YOLO(
            ROOT / "runs/detect/plate_v1/weights/best.pt"
        )
        self.confidence = confidence

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def __call__(self, image_path):
        """Sekil qebul edir, tapilan nomrelerin siyahisini qaytarir."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Sekil oxunmadi: {image_path}"
            )

        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=640,
            device=0,
            verbose=False,
        )[0]

        plates = []

        for detection in result.boxes:
            box = detection.xyxy[0].cpu().tolist()
            confidence = float(detection.conf[0].item())

            crop = crop_with_padding(
                image, box, padding=0.05
            )
            if crop is None:
                continue

            enlarged = cv2.resize(
                crop,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC,
            )
            rgb = cv2.cvtColor(
                enlarged, cv2.COLOR_BGR2RGB
            )

            raw_text = pytesseract.image_to_string(
                rgb,
                lang="eng",
                config="--psm 7",
            )
            text = clean_plate_text(raw_text)

            plates.append({
                "box": box,
                "detector_confidence": confidence,
                "raw_text": raw_text,
                "text": text,
                "status": "ok" if text else "ocr_empty",
            })

        return plates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    pipeline = PlatePipeline()
    plates = pipeline(args.image)

    output = {
        "image": str(args.image),
        "status": "detections_found" if plates else "no_detection",
        "plates": plates,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()