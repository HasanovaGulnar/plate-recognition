from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "benchmarks-master" / "endtoend" / "eu"

line = (DATA / "eu1.txt").read_text(encoding="utf-8").strip()
parts = line.split()

filename = parts[0]
x, y, width, height = map(int, parts[1:5])
plate_text = " ".join(parts[5:])

image = cv2.imread(str(DATA / filename))
if image is None:
    raise FileNotFoundError(f"Sekil oxunmadi: {DATA / filename}")

cv2.rectangle(
    image,
    (x, y),
    (x + width, y + height),
    (0, 255, 0),
    2,
)

cv2.putText(
    image,
    plate_text,
    (x, max(25, y - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

output = ROOT / "results" / "annotation_check.jpg"
output.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output), image):
    raise RuntimeError("Netice sekli yadda saxlanmadi.")

print(f"Nomre: {plate_text}")
print(f"Netice: {output}")