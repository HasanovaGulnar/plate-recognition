from pathlib import Path
import json
import shutil
import cv2

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/benchmarks-master/endtoend/eu"
DEST = ROOT / "data/yolo"

splits = json.loads(
    (ROOT / "data/splits.json").read_text(encoding="utf-8")
)

# Kohne bolgu ile yeni bolgunun qarismasinin qarsisini alir.
if DEST.exists():
    raise FileExistsError(
        "data/yolo artiq movcuddur. Yeniden isletmeden evvel yoxlayaq."
    )

for split, filenames in splits.items():
    image_dir = DEST / "images" / split
    label_dir = DEST / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        source_image = SOURCE / filename
        image = cv2.imread(str(source_image))

        if image is None:
            raise ValueError(f"Sekil oxunmadi: {filename}")

        image_h, image_w = image.shape[:2]
        annotation = source_image.with_suffix(".txt")
        rows = []

        for line in annotation.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if not line.strip():
                continue

            parts = line.split()
            x, y, w, h = map(int, parts[1:5])

            if not (
                0 <= x < x + w <= image_w
                and 0 <= y < y + h <= image_h
            ):
                raise ValueError(f"Yanlis cercive: {filename}")

            center_x = (x + w / 2) / image_w
            center_y = (y + h / 2) / image_h
            norm_w = w / image_w
            norm_h = h / image_h

            rows.append(
                f"0 {center_x:.8f} {center_y:.8f} "
                f"{norm_w:.8f} {norm_h:.8f}"
            )

        if not rows:
            raise ValueError(f"Annotasiya bosdur: {filename}")

        shutil.copy2(source_image, image_dir / filename)
        (label_dir / annotation.name).write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    print(f"{split}: {len(filenames)} sekil ve etiket hazirdir")

config = (
    f"path: {json.dumps(DEST.as_posix())}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: license_plate\n"
)

config_path = DEST / "dataset.yaml"
config_path.write_text(config, encoding="utf-8")
print(f"Konfiqurasiya: {config_path}")