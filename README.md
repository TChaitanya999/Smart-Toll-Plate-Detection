## License Plate Authenticity Detector

This project is a Flask web application that detects a vehicle's license plate from an uploaded image, recognizes the plate text using OCR, and evaluates whether the plate is **real or fake** based on visual consistency checks and a registry lookup.

### Features

- **Object detection**: Uses a YOLO model (`my_model.pt`) to detect license plates in images.
- **OCR ensemble**: Combines EasyOCR and Tesseract-based flows for robust plate text extraction.
- **Post-processing & validation**: Cleans text, fixes common OCR errors, and scores candidates against Indian plate formats.
- **Authenticity checks**: Evaluates spacing, font, and uniformity and cross-references against `plates.csv` via `plate_registry.py`.
- **Web UI**: Single-page frontend (`templates/index.html`) for uploading images and viewing detection, plate text, authenticity decision, and registry details.

### Requirements

- Python 3.9+ (recommended)
- A working C/C++ build toolchain for some dependencies (e.g. `opencv-python`, `easyocr`, `ultralytics`)
- System-level dependencies for Tesseract if you use that path (e.g. `tesseract-ocr` installed on your OS)

Install Python packages with:

```bash
pip install -r requirements.txt
```

### Project Structure

- `app.py` – Main Flask app, routes, and image processing pipeline.
- `detection_functions.py` – Helper functions for OCR, plate reconstruction, scoring, and authenticity checks.
- `plate_registry.py` – Functions to check if a plate is registered and to fetch plate details from `plates.csv`.
- `plates.csv` – CSV registry of known plates and their metadata.
- `init_plate_csv.py` – Utility script to initialize or update the plate registry CSV.
- `templates/index.html` – Frontend UI for image upload and result visualization.
- `tools_strip_comments.py` – Utility to strip comments from code files (dev helper).

### Model & Data Setup

- Place your trained YOLO weights file as `my_model.pt` in the project root (same folder as `app.py`).
- Ensure `plates.csv` exists in the project root. You can:
  - Use the existing file, or
  - Generate one with `init_plate_csv.py` (check the script for schema and usage).

### Running the Application

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server**:

   ```bash
   python app.py
   ```

4. Open a browser and go to:

   ```text
   http://127.0.0.1:5000/
   ```

   Upload an image containing a vehicle license plate to see the detection, recognized plate, authenticity result, and any matched registry record.

### Notes & Troubleshooting

- If YOLO or EasyOCR fail to load, double-check GPU/CPU compatibility and that the versions in `requirements.txt` are installed correctly.
- If Tesseract-related functions fail, make sure the Tesseract executable is installed and added to your system `PATH`.
- For large images, the app resizes them internally for inference; quality issues or extreme angles may still degrade OCR performance.

