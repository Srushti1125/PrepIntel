"""PDF discovery and page-by-page text extraction (PyMuPDF) with scanned-page detection + optional OCR."""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)


@dataclass
class PageText:
    page: int          # 1-based, matches the PDF viewer
    text: str
    ocr: bool = False


@dataclass
class LoadedPDF:
    path: Path
    sha256: str
    pages: list[PageText] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)   # image-only pages (no text layer)
    ocr_used: bool = False
    error: str | None = None

    @property
    def has_text(self) -> bool:
        return any(p.text.strip() for p in self.pages)


def discover_pdfs(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ocr_page(page: "pymupdf.Page") -> str:
    """OCR one page with Tesseract. Returns '' (and logs) if OCR is unavailable."""
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception as e:  # missing binary, missing package, ...
        log.warning("OCR unavailable or failed: %s", e)
        return ""


def load_pdf(path: Path, ocr_enabled: bool = False, min_chars: int = 20) -> LoadedPDF:
    path = Path(path)
    sha = file_sha256(path)
    result = LoadedPDF(path=path, sha256=sha)
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        result.error = f"cannot open PDF: {e}"
        return result
    with doc:
        if doc.needs_pass:
            result.error = "PDF is password protected"
            return result
        for number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            used_ocr = False
            if len(text) < min_chars and page.get_images(full=True):
                result.scanned_pages.append(number)
                if ocr_enabled:
                    ocr_text = clean_text(_ocr_page(page))
                    if ocr_text:
                        text, used_ocr = ocr_text, True
                        result.ocr_used = True
            result.pages.append(PageText(page=number, text=text, ocr=used_ocr))
    return result
