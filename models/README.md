# PlateGuard model and inference API

Pipeline: OpenCV image → YOLO plate detection → crop → preprocessing → PaddleOCR → plate text, confidences, and box coordinates.

## Trained weights

Put the trained `best.pt` file at `models/weights/best.pt`. This folder is
Git-ignored; share the weights separately or use Git LFS if the team requires
them in Git. Alternatively, set `PLATEGUARD_WEIGHTS` to the file's path.

## Run the inference API

From the `models` directory, install the model requirements and start the API:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The server loads YOLO and PaddleOCR once at startup. By default PaddleOCR uses
CPU; set `PLATEGUARD_OCR_DEVICE=gpu:0` to select an available Paddle GPU.

Check `http://localhost:8000/health`. Send a JPEG, PNG, or WebP image as the
`file` form field to `POST http://localhost:8000/api/predict`. Uploads are
limited to 10 MB and 20 megapixels. Each item in `plates` includes the
registration text (or `null` if OCR could not read it), detector and OCR
confidence, and the original-image bounding box. The endpoint returns an
empty `plates` list when no plate is detected.

## Tests

From the `models` directory:

```powershell
python -m pytest tests
```
