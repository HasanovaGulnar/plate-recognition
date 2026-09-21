from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"

images = sorted(
    p for p in DATA.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

valid = 0
missing = []
invalid = []
multiple = []

for path in images:
    annotation = path.with_suffix(".txt")

    if not annotation.exists():
        missing.append(path.name)
        continue

    try:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Sekil oxunmur")

        image_h, image_w = image.shape[:2]
        lines = annotation.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        lines = [line for line in lines if line.strip()]

        if not lines:
            raise ValueError("Annotasiya bosdur")

        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                raise ValueError("Annotasiya formati uygun deyil")

            if parts[0] != path.name:
                raise ValueError("Sekil adi uygun deyil")

            x, y, w, h = map(int, parts[1:5])

            if not (
                x >= 0 and y >= 0 and w > 0 and h > 0
                and x + w <= image_w
                and y + h <= image_h
            ):
                raise ValueError("Cercive sekil serhedlerine uygun deyil")

        valid += 1
        if len(lines) > 1:
            multiple.append(path.name)

    except (ValueError, UnicodeError) as error:
        invalid.append(f"{path.name}: {error}")

print(f"Umumi sekil: {len(images)}")
print(f"Yoxlamadan kecen: {valid}")
print(f"TXT olmayan: {len(missing)}")
print(f"Problemli: {len(invalid)}")
print(f"Bir nece nomre annotasiyasi olan: {len(multiple)}")

for item in missing:
    print("TXT YOXDUR:", item)

for item in invalid:
    print("PROBLEM:", item)

for item in multiple:
    print("COX NOMRE:", item)