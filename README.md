# License Plate Detection and Recognition

A Python pipeline that detects license plates with YOLOv8n,
crops them with 5% padding, and reads their text using Tesseract.

## Pipeline

Image → YOLOv8n → padded crop → Tesseract PSM 7 → text cleaning.

Text cleaning converts output to uppercase and removes non-alphanumeric
characters. Raw OCR output is preserved. All detection boxes come
from the model.

## Dataset

Source: https://github.com/openalpr/benchmarks

108 EU images: 73 train, 15 validation, and 20 test.
Images sharing the same annotated plate text are grouped together.
Existing dataset annotations are used.

## Test Results

- Mean IoU: 0.888
- Detection precision: 90.9%
- Detection recall: 100.0%
- End-to-end exact match: 15.0% (3/20)
- End-to-end character error rate: 46.1%

Detection matching uses IoU ≥ 0.5. Missing detections count as empty
text predictions. Some background plates are unannotated, so detection
metrics describe annotated targets rather than every visible plate.

## OCR Comparison

On the same 15 validation crops:

- EasyOCR: 13.3% exact match, 22.1% CER.
- Tesseract PSM 7: 40.0% exact match, 31.7% CER.
- Tesseract PSM 13: 20.0% exact match, 34.6% CER.

PSM 7 was selected for its higher exact-match accuracy.

## Setup and Usage

Install Python dependencies:

python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

Install Tesseract separately with English language data.
The configured Windows path is:

    C:\Program Files\Tesseract-OCR\tesseract.exe

The trained model is included at:

    runs/detect/plate_v1/weights/best.pt

Run the pipeline:

    python pipeline.py path/to/image.jpg

Evaluate the test set:

    python evaluate_test.py

The current configuration requires a CUDA-compatible NVIDIA GPU.
Evaluation also requires the dataset and data/splits.json.

## Outputs

- results/test_report.csv — per-image predictions and correctness.
- results/test_summary.json — aggregate metrics.
- results/failures/ — three illustrated OCR failures.
- results/multi_vehicle/ — additional multi-vehicle checks.

## Failure Analysis and Limitations

- eu6.jpg: angled, blurry plate; multiple OCR character errors.
- eu10.jpg: extra "NE" prefix, possibly caused by non-text plate elements.
- test_006.jpg: 9/S confusion and a possible O/0 annotation ambiguity.

Three multi-vehicle training images were checked. Foreground crops
were intact, but background plates were missed. These checks are
not independent test results.

A synthetic blank image correctly returned no detections.
OCR accuracy remains low, and deskewing is not implemented.
The pipeline is a learning prototype, not production-ready.