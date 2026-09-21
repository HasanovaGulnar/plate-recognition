from pathlib import Path
from collections import defaultdict
import hashlib
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"

groups = defaultdict(list)

for path in sorted(DATA.iterdir()):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Sekil oxunmadi: {path.name}")

    digest = hashlib.sha256(
        str(image.shape).encode() + image.tobytes()
    ).hexdigest()

    groups[digest].append(path.name)

duplicates = [names for names in groups.values() if len(names) > 1]

print(f"Unikal sekil: {len(groups)}")
print(f"Tekrar qruplari: {len(duplicates)}")

for names in duplicates:
    print("TEKRAR:", ", ".join(names))