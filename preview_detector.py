from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    model = YOLO(
        ROOT / "runs/detect/plate_v1/weights/best.pt"
    )

    results = model.predict(
        source=str(ROOT / "data/yolo/images/val"),
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        project=str(ROOT / "results"),
        name="detector_preview",
        exist_ok=True,
    )

    for result in results:
        print(
            f"{Path(result.path).name}: "
            f"{len(result.boxes)} nomre tapildi"
        )

    print(f"Sekiller: {ROOT / 'results/detector_preview'}")


if __name__ == "__main__":
    main()