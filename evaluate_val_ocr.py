from pathlib import Path
import json
from clean_text import clean_plate_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current

    return previous[-1]


def main():
    splits = json.loads(
        (ROOT / "data/splits.json").read_text(encoding="utf-8")
    )
    records = json.loads(
        (ROOT / "results/ocr_cleaned.json").read_text(
            encoding="utf-8"
        )
    )

    correct = 0
    total_errors = 0
    total_chars = 0
    missing = 0

    for filename in splits["val"]:
        parts = (DATA / Path(filename).with_suffix(".txt")).read_text(
            encoding="utf-8-sig"
        ).split()
        expected = clean_plate_text("".join(parts[5:]))

        prefix = f"{Path(filename).stem}_plate_"
        matches = [
            record for record in records
            if record["crop"].startswith(prefix)
        ]

        if len(matches) > 1:
            raise ValueError(
                f"{filename}: bir nece crop var; "
                "evvel cerciveleri uygunlasdirmaq lazimdir."
            )

        predicted = matches[0]["cleaned_text"] if matches else ""
        if not matches:
            missing += 1

        matched = predicted == expected
        correct += int(matched)
        total_errors += edit_distance(expected, predicted)
        total_chars += len(expected)

        status = "DUZGUN" if matched else "SEHV"
        print(
            f"{filename}: gozlenen={expected} | "
            f"OCR={predicted or '[BOS]'} | {status}"
        )

    count = len(splits["val"])
    print(f"\nTam duzgun: {correct}/{count}")
    print(f"Exact-match: {100 * correct / count:.1f}%")
    print(f"CER: {100 * total_errors / total_chars:.1f}%")
    print(f"Crop olmayan sekil: {missing}")


if __name__ == "__main__":
    main()