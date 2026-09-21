from pathlib import Path
import json
import cv2
import pytesseract

ROOT = Path(__file__).resolve().parent
CONFIG = "--psm 13"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def main():
    print(f"Isleyen OCR rejimi: {CONFIG}", flush=True)

    crops = sorted((ROOT / "results/crops").glob("*.png"))
    if not crops:
        raise FileNotFoundError("Crop sekilleri tapilmadi.")

    records = []

    for path in crops:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Sekil oxunmadi: {path.name}")

        enlarged = cv2.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)

        raw_text = pytesseract.image_to_string(
            rgb,
            lang="eng",
            config=CONFIG,
        )

        records.append({
            "crop": path.name,
            "raw_text": raw_text,
            "engine": "tesseract",
            "config": CONFIG,
            "ocr_image_scale": 2,
        })

        print(f"{path.name} -> {raw_text.strip() or '[BOS]'}")

    content = json.dumps(
        records,
        indent=2,
        ensure_ascii=False,
    )

    for filename in (
        "ocr_raw_tesseract_psm13.json",
        "ocr_raw.json",
    ):
        (ROOT / "results" / filename).write_text(
            content,
            encoding="utf-8",
        )

    print(f"Tamamlandi. Saxlanilan rejim: {CONFIG}")


if __name__ == "__main__":
    main()