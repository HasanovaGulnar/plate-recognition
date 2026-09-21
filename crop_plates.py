from pathlib import Path
import math
from ultralytics import YOLO
import cv2

ROOT = Path(__file__).resolve().parent


def crop_with_padding(image, box, padding=0.05):
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    left = max(0, math.floor(x1 - pad_x))
    top = max(0, math.floor(y1 - pad_y))
    right = min(image_w, math.ceil(x2 + pad_x))
    bottom = min(image_h, math.ceil(y2 + pad_y))

    if right <= left or bottom <= top:
        return None

    return image[top:bottom, left:right].copy()


def main():
    model = YOLO(
        ROOT / "runs/detect/plate_v1/weights/best.pt"
    )

    output = ROOT / "results/crops"
    output.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(ROOT / "data/yolo/images/val"),
        conf=0.25,
        imgsz=640,
        device=0,
        stream=True,
        verbose=False,
    )

    total = 0

    for result in results:
        filename = Path(result.path).stem

        if len(result.boxes) == 0:
            print(f"{filename}: nomre tapilmadi")
            continue

        for index, box in enumerate(
            result.boxes.xyxy.cpu().tolist(), start=1
        ):
            crop = crop_with_padding(result.orig_img, box)

            if crop is None:
                print(f"{filename}: kecersiz cercive")
                continue

            destination = output / f"{filename}_plate_{index}.png"

            if not cv2.imwrite(str(destination), crop):
                raise RuntimeError(f"Saxlanmadi: {destination}")

            total += 1
            print(f"Saxlanildi: {destination.name}")

    print(f"Umumi kesilmis nomre: {total}")


if __name__ == "__main__":
    main()