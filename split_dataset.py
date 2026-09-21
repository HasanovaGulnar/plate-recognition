from pathlib import Path
from collections import defaultdict
import json
import random
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/benchmarks-master/endtoend/eu"
OUTPUT = ROOT / "data/splits.json"

groups = defaultdict(list)

for path in sorted(DATA.iterdir()):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    lines = [
        line for line in path.with_suffix(".txt")
        .read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise ValueError(f"Bir annotasiya gozlenilirdi: {path.name}")

    text = "".join(lines[0].split()[5:])
    plate = re.sub(r"[^A-Z0-9]", "", text.upper())

    if not plate:
        raise ValueError(f"Nomre metni bosdur: {path.name}")

    groups[plate].append(path.name)

# Sabit seed: tekrar isledende eyni bolgu alinir.
plate_ids = sorted(groups)
random.Random(42).shuffle(plate_ids)


def take_groups(min_images):
    selected = []
    while plate_ids and len(selected) < min_images:
        plate = plate_ids.pop()
        selected.extend(groups[plate])
    return sorted(selected)


test = take_groups(20)
val = take_groups(15)
train = sorted(
    filename
    for plate in plate_ids
    for filename in groups[plate]
)

if len(test) < 20 or len(val) < 15 or not train:
    raise ValueError("Bu bolgu ucun kifayet qeder musteqil qrup yoxdur.")

splits = {"train": train, "val": val, "test": test}

# Her sekil yalniz bir destde olmalidir.
all_names = train + val + test
assert len(all_names) == len(set(all_names))
assert len(all_names) == sum(len(items) for items in groups.values())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(splits, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Unikal nomre qrupu: {len(groups)}")
for name, files in splits.items():
    print(f"{name}: {len(files)} sekil")
print(f"Bolgu saxlanildi: {OUTPUT}")