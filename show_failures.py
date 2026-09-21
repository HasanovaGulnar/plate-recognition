from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

from crop_plates import crop_with_padding

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"


def fit_image(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_CUBIC,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main():
    report = ROOT / "results/test_report.csv"
    with report.open(encoding="utf-8-sig", newline="") as file:
        failures = [
            row for row in csv.DictReader(file)
            if row["correct"].lower() == "false"
        ][:3]

    output = ROOT / "results/failures"
    output.mkdir(parents=True, exist_ok=True)

    for row in failures:
        image = cv2.imread(str(DATA / row["image"]))
        if image is None:
            raise FileNotFoundError(row["image"])

        if not row["matched_box"]:
            raise ValueError(
                f"{row['image']}: uygun detection yoxdur."
            )

        box = json.loads(row["matched_box"])
        crop = crop_with_padding(image, box, padding=0.05)

        if crop is None:
            raise ValueError(f"Kecersiz crop: {row['image']}")

        annotated = image.copy()
        x1, y1, x2, y2 = box

        cv2.rectangle(
            annotated,
            (math.floor(x1), math.floor(y1)),
            (math.ceil(x2), math.ceil(y2)),
            (0, 200, 0),
            2,
        )

        canvas = np.full((520, 1100, 3), 245, dtype=np.uint8)
        canvas[60:460, 20:620] = fit_image(annotated, 600, 400)
        canvas[110:330, 650:1080] = fit_image(crop, 430, 220)

        labels = [
            (row["image"], (20, 35)),
            ("Model detection", (20, 490)),
            ("Crop with 5% padding", (650, 85)),
            (f"GT: {row['expected_text']}", (650, 375)),
            (f"OCR: {row['predicted_text'] or '[EMPTY]'}", (650, 410)),
            (f"IoU: {float(row['best_iou']):.3f}", (650, 445)),
        ]

        for text, position in labels:
            cv2.putText(
                canvas, text, position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (25, 25, 25), 2, cv2.LINE_AA,
            )

        destination = output / f"{Path(row['image']).stem}_failure.jpg"
        if not cv2.imwrite(str(destination), canvas):
            raise RuntimeError(f"Saxlanmadi: {destination}")

        print(f"Saxlanildi: {destination.name}")


if __name__ == "__main__":
    main()