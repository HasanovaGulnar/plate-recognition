from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent


def clean_plate_text(raw_text):
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper())


def main():
    source = ROOT / "results/ocr_raw.json"
    records = json.loads(source.read_text(encoding="utf-8"))

    for record in records:
        raw = record["raw_text"]
        cleaned = clean_plate_text(raw)
        record["cleaned_text"] = cleaned

        print(f"{record['crop']}: {raw!r} -> {cleaned!r}")

    output = ROOT / "results/ocr_cleaned.json"
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saxlanildi: {output}")


if __name__ == "__main__":
    main()