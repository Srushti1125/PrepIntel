"""CLI: python scripts/ingest.py [--input-dir DIR] [--force] [--reset] [--skip-insights] [--ocr]"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import load_settings  # noqa: E402
from ingestion.pipeline import IngestionError, run_ingestion  # noqa: E402
from ingestion.vector_store import get_embeddings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest interview-experience PDFs into ChromaDB + JSON.")
    ap.add_argument("--input-dir", help="override PDF_INPUT_DIR")
    ap.add_argument("--force", action="store_true", help="re-process files even if unchanged")
    ap.add_argument("--reset", action="store_true", help="wipe index, JSON and manifest first")
    ap.add_argument("--skip-insights", action="store_true", help="index only; no LLM extraction")
    ap.add_argument("--ocr", action="store_true", help="OCR scanned pages (needs Tesseract)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = load_settings()
    if args.input_dir:
        settings = dataclasses.replace(settings, pdf_input_dir=Path(args.input_dir).resolve())

    llm = None
    if not args.skip_insights:
        try:
            from backend.services import get_llm

            llm = get_llm(settings)
        except Exception as e:
            print(f"! LLM unavailable ({e}). Continuing with --skip-insights behaviour; dashboard data will be empty.")

    print(f"Loading embedding model {settings.embedding_model} (CPU) ...")
    embeddings = get_embeddings(settings.embedding_model)
    try:
        report = run_ingestion(settings, embeddings, llm, force=args.force, reset=args.reset,
                               skip_insights=args.skip_insights, ocr=args.ocr or None)
    except IngestionError as e:
        print(f"ERROR: {e}")
        return 2

    print("\n=== Ingestion summary ===")
    print(f"indexed: {len(report.indexed)}  skipped(unchanged): {len(report.skipped)}  "
          f"duplicates: {len(report.duplicates)}  failed: {len(report.failed)}")
    print(f"new chunks: {report.chunks}  structured experiences: {report.experiences}")
    for sp, err in {**report.failed, **{k: f'insights: {v}' for k, v in report.insight_failures.items()}}.items():
        print(f"  ! {sp}: {err}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
