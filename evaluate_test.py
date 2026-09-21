from pathlib import Path
import csv
import json

from pipeline import PlatePipeline
from clean_text import clean_plate_text
from evaluate_val_ocr import edit_distance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"


def calculate_iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def main():
    splits = json.loads(
        (ROOT / "data/splits.json").read_text(encoding="utf-8")
    )
    filenames = splits["test"]

    if len(filenames) < 20:
        raise ValueError("En azi 20 test sekli lazimdir.")

    pipeline = PlatePipeline(confidence=0.25)

    rows = []
    ious = []
    true_positive = 0
    false_positive = 0
    correct = 0
    character_errors = 0
    character_count = 0

    for filename in filenames:
        annotation = DATA / Path(filename).with_suffix(".txt")
        lines = [
            line for line in annotation.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        # Bu test destinde her sekilde bir etiket var.
        if len(lines) != 1:
            raise ValueError(
                f"{filename}: bir ground-truth gozlenilirdi."
            )

        parts = lines[0].split()
        x, y, width, height = map(int, parts[1:5])
        ground_truth_box = [x, y, x + width, y + height]
        expected = clean_plate_text("".join(parts[5:]))

        predictions = pipeline(DATA / filename)
        overlaps = [
            calculate_iou(item["box"], ground_truth_box)
            for item in predictions
        ]

        best_index = (
            max(range(len(overlaps)), key=overlaps.__getitem__)
            if overlaps else None
        )
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        detected = best_iou >= 0.5

        # Yalniz IoU >= 0.5 olan proqnoz GT ile uygunlasdirilir.
        matched = predictions[best_index] if detected else None
        predicted_text = matched["text"] if matched else ""

        true_positive += int(detected)
        false_positive += len(predictions) - int(detected)
        ious.append(best_iou)

        exact_match = detected and predicted_text == expected
        correct += int(exact_match)
        character_errors += edit_distance(expected, predicted_text)
        character_count += len(expected)

        rows.append({
            "image": filename,
            "ground_truth_box": json.dumps(ground_truth_box),
            "expected_text": expected,
            "predictions": json.dumps(predictions, ensure_ascii=False),
            "best_iou": round(best_iou, 6),
            "matched_box": json.dumps(matched["box"]) if matched else "",
            "raw_text": matched["raw_text"] if matched else "",
            "predicted_text": predicted_text,
            "correct": exact_match,
        })

        print(
            f"{filename}: IoU={best_iou:.3f} | "
            f"GT={expected} | OCR={predicted_text or '[BOS]'} | "
            f"{'DUZGUN' if exact_match else 'SEHV'}"
        )

    count = len(filenames)
    false_negative = count - true_positive
    precision = true_positive / max(1, true_positive + false_positive)

    summary = {
        "test_images": count,
        "iou_threshold": 0.5,
        "mean_iou": sum(ious) / count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "detection_precision": precision,
        "detection_recall": true_positive / count,
        "end_to_end_exact_match": correct / count,
        "end_to_end_cer": character_errors / character_count,
    }

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)

    with (output / "test_report.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    (output / "test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nTest sekilleri: {count}")
    print(f"Orta IoU: {summary['mean_iou']:.3f}")
    print(f"Detection precision: {100 * precision:.1f}%")
    print(f"Detection recall: {100 * true_positive / count:.1f}%")
    print(f"End-to-end exact-match: {100 * correct / count:.1f}%")
    print(f"End-to-end CER: {100 * character_errors / character_count:.1f}%")
    print("Hesabatlar results qovlugunda saxlanildi.")


if __name__ == "__main__":
    main()