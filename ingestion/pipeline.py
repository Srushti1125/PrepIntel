"""Ingestion orchestration: PDFs -> chunks -> ChromaDB, PDFs -> structured JSON, manifest tracking."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ingestion.insight_extractor import extract_experiences, rebuild_companies_index, save_records
from ingestion.metadata_extractor import MetadataError, extract_metadata
from ingestion.pdf_loader import discover_pdfs, file_sha256, load_pdf
from ingestion.text_splitter import build_chunks, chunk_ids, redact_pii
from ingestion.vector_store import add_chunks, delete_source, embedding_dimension, get_vector_store


def text_fingerprint(pages) -> str:
    """Hash of the normalised extracted text: catches the same document re-exported as a different PDF."""
    import hashlib
    import re

    norm = re.sub(r"\s+", " ", " ".join(p.text for p in pages)).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class IngestionError(RuntimeError):
    pass


@dataclass
class IngestionReport:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    insight_failures: dict[str, str] = field(default_factory=dict)
    experiences: int = 0
    chunks: int = 0


def manifest_path(data_dir: Path) -> Path:
    return Path(data_dir) / "ingestion_manifest.json"


def load_manifest(data_dir: Path) -> dict:
    p = manifest_path(data_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "embedding": {}, "documents": {}}


def save_manifest(data_dir: Path, manifest: dict) -> None:
    p = manifest_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def run_ingestion(
    settings,
    embeddings,
    llm=None,
    *,
    force: bool = False,
    reset: bool = False,
    skip_insights: bool = False,
    ocr: bool | None = None,
    progress: Callable[[str], None] = print,
) -> IngestionReport:
    data_dir, root = Path(settings.data_dir), Path(settings.pdf_input_dir)
    if not root.exists():
        raise IngestionError(f"PDF input directory not found: {root}")
    ocr_enabled = settings.ocr_enabled if ocr is None else ocr
    want_insights = llm is not None and not skip_insights

    if reset:
        progress("Resetting vector store, structured data and manifest ...")
        try:
            get_vector_store(settings, embeddings).delete_collection()
        except Exception:
            pass
        shutil.rmtree(data_dir / "experiences", ignore_errors=True)
        for f in (manifest_path(data_dir), data_dir / "companies.json"):
            f.unlink(missing_ok=True)

    manifest = load_manifest(data_dir)
    dim = embedding_dimension(embeddings)
    prev = manifest.get("embedding") or {}
    if prev and (prev.get("model") != settings.embedding_model or prev.get("dim") != dim):
        raise IngestionError(
            f"Index was built with {prev.get('model')} (dim {prev.get('dim')}) but the current model is "
            f"{settings.embedding_model} (dim {dim}). Re-run with --reset to rebuild the index."
        )
    manifest["embedding"] = {"model": settings.embedding_model, "dim": dim}
    store = get_vector_store(settings, embeddings)
    docs = manifest["documents"]
    report = IngestionReport()

    pdfs = discover_pdfs(root)
    progress(f"Found {len(pdfs)} PDF(s) under {root}")
    for path in pdfs:
        try:
            meta = extract_metadata(path, root)
        except MetadataError as e:
            rel = path.relative_to(root).as_posix()
            report.failed[rel] = str(e)
            docs[rel] = {"status": "failed", "error": str(e), "processed_at": time.time()}
            continue

        sp, sha = meta.source_path, file_sha256(path)
        old = docs.get(sp)
        unchanged = bool(old) and old.get("sha256") == sha and not force
        if unchanged and old["status"] == "duplicate":
            report.skipped.append(sp)
            continue
        if unchanged and old["status"] == "indexed" and (not want_insights or old.get("insights")):
            report.skipped.append(sp)
            continue
        need_index = not (unchanged and old["status"] == "indexed")

        loaded = load_pdf(path, ocr_enabled=ocr_enabled)
        base = {"sha256": sha, "company": meta.company, "company_id": meta.company_id, "year": meta.year,
                "filename": meta.filename, "pages": len(loaded.pages), "scanned_pages": loaded.scanned_pages,
                "ocr_used": loaded.ocr_used, "processed_at": time.time()}
        if loaded.error or not loaded.has_text:
            msg = loaded.error or (
                "no extractable text" + (" (scanned/image-only pages detected; enable OCR with --ocr)" if loaded.scanned_pages else "")
            )
            docs[sp] = {**base, "status": "failed", "error": msg}
            report.failed[sp] = msg
            progress(f"  FAILED     {sp}: {msg}")
            save_manifest(data_dir, manifest)
            continue
        for p in loaded.pages:
            p.text = redact_pii(p.text)
        tsha = text_fingerprint(loaded.pages)
        base["text_sha256"] = tsha
        if need_index:
            dup = next((o for o, d in docs.items() if o != sp and d.get("status") == "indexed"
                        and (d.get("sha256") == sha or d.get("text_sha256") == tsha)), None)
            if dup:
                docs[sp] = {**base, "status": "duplicate", "duplicate_of": dup}
                report.duplicates.append(sp)
                progress(f"  duplicate  {sp}  (same content as {dup})")
                save_manifest(data_dir, manifest)
                continue

        try:
            entry = dict(old) if unchanged else {}
            entry.update(base)
            if need_index:
                delete_source(store, sp)
                chunks = build_chunks(loaded.pages, meta, sha, settings.chunk_size, settings.chunk_overlap)
                add_chunks(store, chunks, chunk_ids(sha, len(chunks)))
                entry.update(status="indexed", chunks=len(chunks), insights=False, experiences=0)
                entry.pop("error", None)
                report.chunks += len(chunks)
            report.indexed.append(sp)
            progress(f"  indexed    {sp}" + (f"  ({entry['chunks']} chunks)" if need_index else "  (insights only)"))
        except Exception as e:
            docs[sp] = {**base, "status": "failed", "error": f"indexing failed: {e}"}
            report.failed[sp] = f"indexing failed: {e}"
            progress(f"  FAILED     {sp}: {e}")
            save_manifest(data_dir, manifest)
            continue

        if want_insights:
            try:
                records = extract_experiences(llm, loaded.pages, meta, sha)
                save_records(data_dir, meta, records)
                entry.update(insights=True, experiences=len(records))
                entry.pop("insights_error", None)
                report.experiences += len(records)
                progress(f"             -> {len(records)} structured experience(s)")
            except Exception as e:
                entry.update(insights=False, insights_error=str(e))
                report.insight_failures[sp] = str(e)
                progress(f"             insight extraction failed: {e}")
        docs[sp] = entry
        save_manifest(data_dir, manifest)

    save_manifest(data_dir, manifest)
    rebuild_companies_index(data_dir, manifest)
    return report
