from pathlib import Path
import json
import cv2

from pipeline import PlatePipeline
from crop_plates import crop_with_padding

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Saxlanmadi: {path}")


def main():
    filenames = [
        "test_055.jpg",
        "test_056.jpg",
        "test_082.jpg",
    ]

    output = ROOT / "results/multi_vehicle"
    output.mkdir(parents=True, exist_ok=True)

    pipeline = PlatePipeline()
    records = []

    splits = json.loads(
        (ROOT / "data/splits.json").read_text(encoding="utf-8")
    )

    for filename in filenames:
        path = DATA / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(str(path))

        plates = pipeline(path)
        annotated = image.copy()
        crop_files = []

        for index, plate in enumerate(plates, start=1):
            box = plate["box"]
            crop = crop_with_padding(image, box, padding=0.05)

            if crop is None:
                raise ValueError(f"Kecersiz crop: {filename}")

            crop_name = f"{path.stem}_plate_{index}.png"
            save_image(output / crop_name, crop)
            crop_files.append(crop_name)

            x1, y1, x2, y2 = [round(value) for value in box]
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 255, 0), 2,
            )

            label = f"{index}: {plate['text'] or '[EMPTY]'}"
            cv2.putText(
                annotated,
                label,
                (max(0, x1), max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        save_image(
            output / f"{path.stem}_detections.jpg",
            annotated,
        )

        source_split = next(
            (
                name for name, items in splits.items()
                if filename in items
            ),
            "unknown",
        )

        records.append({
            "image": filename,
            "source_split": source_split,
            "visible_plates_visual_review": 2,
            "detection_count": len(plates),
            "plates": plates,
            "crop_files": crop_files,
            "note": (
                "Supplementary visual check. "
                "Detection count alone does not verify correct matching."
            ),
        })

        print(
            f"{filename}: {len(plates)} detection, "
            f"{len(crop_files)} crop | dest: {source_split}"
        )

    (output / "report.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Neticeler: {output}")


if __name__ == "__main__":
    main()