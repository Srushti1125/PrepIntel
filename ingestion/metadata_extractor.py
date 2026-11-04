"""Derive company / year / filename from the existing folder hierarchy (never from the LLM)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from schemas.experience import make_company_id


class MetadataError(ValueError):
    pass


@dataclass(frozen=True)
class DocMeta:
    company: str        
    company_id: str     
    year: str
    filename: str
    source_path: str    


def extract_metadata(pdf_path: Path, root: Path) -> DocMeta:
    pdf_path, root = Path(pdf_path), Path(root)
    try:
        rel = pdf_path.resolve().relative_to(root.resolve())
    except ValueError as e:
        raise MetadataError(f"{pdf_path} is not inside {root}") from e
    parts = rel.parts
    if len(parts) < 3:
        raise MetadataError(f"{rel.as_posix()}: expected <year>/<company>/<file>.pdf")
    year = parts[0]
    if not re.fullmatch(r"\d{4}", year):
        raise MetadataError(f"{rel.as_posix()}: first folder '{year}' is not a 4-digit year")
    company = pdf_path.parent.name  # immediate parent directory
    return DocMeta(
        company=company,
        company_id=make_company_id(company),
        year=year,
        filename=pdf_path.name,
        source_path=rel.as_posix(),
    )
