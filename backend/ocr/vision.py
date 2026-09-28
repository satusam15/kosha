"""Extracts text from images and scanned PDFs.

Primary path: Tesseract (via pytesseract) — a classical, purely local OCR
engine. It reliably transcribes printed/typed text, which is what scanned
inspection reports and forms are, unlike small vision-LLMs (moondream is a
1.6B captioning/VQA model — testing showed it hallucinates and drops key
values like pressure readings on dense text, rather than transcribing them).

Fallback path: the local Moondream vision model via Ollama, used only when
Tesseract finds no meaningful text — i.e. the input looks like a genuine
photo/diagram rather than a scanned document.

All local — talks to Ollama at `http://localhost:11434` only; Tesseract runs
as a local binary with no network calls at all.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import ollama
import pytesseract
from PIL import Image

# TODO: use the same vision model name as the router's VISION_MODEL.
VISION_MODEL = "moondream"              # TODO: your local vision model name
OLLAMA_BASE_URL = "http://localhost:11434"

_client = ollama.Client(host=OLLAMA_BASE_URL)

# A small set of prompts used for the vision-model fallback.
_DEFAULT_PROMPT = "Describe what is shown in this image in detail."
_TABLE_PROMPT = "Extract the text in this image as a markdown table if it looks tabular, otherwise transcribe verbatim."

# Below this many non-whitespace characters, Tesseract likely found nothing
# useful (e.g. a photo/diagram rather than a text document) -> fall back.
_MIN_TESSERACT_CHARS = 8


def _image_to_base64_bytes(image_path: Path) -> bytes:
    """Read an image and return its raw bytes (PNG/JPEG supported by Ollama)."""
    data = image_path.read_bytes()
    # base64 encode for the Ollama image payload.
    return base64.b64encode(data)


def _vision_fallback(image: Image.Image, prompt: str) -> str:
    """Ask the local vision model to describe an image Tesseract couldn't read."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    response = _client.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
    )
    return response["message"]["content"].strip()


def _ocr_pil_image(image: Image.Image, prompt: str) -> str:
    """Try Tesseract first; fall back to the vision model for non-text images."""
    text = pytesseract.image_to_string(image).strip()
    if len(text.replace("\n", "").replace(" ", "")) >= _MIN_TESSERACT_CHARS:
        return text
    return f"[vision-fallback, no OCR text detected] {_vision_fallback(image, prompt)}"


def ocr_image(image_path: Path | str, prompt: str = _DEFAULT_PROMPT) -> str:
    """Extract text from a single image (Tesseract, with a vision-model fallback)."""
    image_path = Path(image_path)
    if not image_path.exists():
        return f"[ocr] image not found: {image_path}"

    image = Image.open(image_path)
    return _ocr_pil_image(image, prompt)


def ocr_pdf(pdf_path: Path | str, prompt: str = _DEFAULT_PROMPT) -> str:
    """Rasterize each page of a PDF to an image and OCR them in sequence.

    Uses `pdf2image` (poppler) locally.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return f"[ocr] pdf not found: {pdf_path}"

    try:
        from pdf2image import convert_from_path
    except ImportError:
        return "[ocr] pdf2image not installed; run: pip install pdf2image"

    try:
        pages = convert_from_path(str(pdf_path), dpi=200)
    except Exception as exc:  # noqa: BLE001
        return f"[ocr] could not rasterize PDF (is poppler installed?): {exc}"

    out: list[str] = []
    for page_no, img in enumerate(pages, start=1):
        text = _ocr_pil_image(img, prompt)
        out.append(f"--- page {page_no} ---\n{text}")

    return "\n\n".join(out)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("usage: python -m backend.ocr.vision <image.png|scan.pdf>")
        sys.exit(1)
    p = Path(target)
    if p.suffix.lower() == ".pdf":
        print(ocr_pdf(p))
    else:
        print(ocr_image(p))
