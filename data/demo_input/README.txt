Placeholder for sample scanned inspection reports.

Put a scanned PDF or image here and OCR it with the vision model:

    python -m backend.ocr.vision data/demo_input/inspection_report_scan.pdf

or via the API:

    curl "http://127.0.0.1:8000/ocr?path=/abs/path/to/file.png"

The OCR pipeline rasterizes each page (poppler via `pdf2image`) and sends it
to your local Moondream / Qwen2-VL model in Ollama. No data leaves the box.

TODO: add a real scanned inspection report (e.g. inspection_report_scan.pdf,
      equipment_tag_photo.jpg).
