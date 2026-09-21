from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # Ilk defe isledende ilkin model avtomatik yuklenir.
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(ROOT / "data/yolo/dataset.yaml"),
        epochs=80,
        patience=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        seed=42,
        fliplr=0.0,
        flipud=0.0,
        project=str(ROOT / "runs/detect"),
        name="plate_v1",
        plots=True,
    )

    print(f"En yaxsi model: {model.trainer.best}")


if __name__ == "__main__":
    main()