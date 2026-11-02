"""Pydantic models.

* ``LLM*`` models are what the LLM is asked to return (free text, nothing about company/year).
* ``ExperienceRecord`` and friends are the validated, grounded records stored as JSON.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


def make_company_id(name: str) -> str:
    """Canonical, filesystem-safe company identifier."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def display_name(folder_name: str) -> str:
    return re.sub(r"\s+", " ", folder_name.replace("_", " ")).strip()


# --------------------------------------------------------------------------- LLM output schema
class LLMQuestion(BaseModel):
    text: str
    kind: str = "other"
    page: Optional[int] = None


class LLMRound(BaseModel):
    name: str
    category: str = "Other"
    description: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    questions: list[LLMQuestion] = Field(default_factory=list)
    difficulty: Optional[str] = None
    pages: list[int] = Field(default_factory=list)


class LLMTip(BaseModel):
    text: str
    category: str = "Interview strategy"
    page: Optional[int] = None


class LLMResource(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    topic: Optional[str] = None
    page: Optional[int] = None


class LLMExperience(BaseModel):
    role: Optional[str] = None
    rounds: list[LLMRound] = Field(default_factory=list)
    tips: list[LLMTip] = Field(default_factory=list)
    resources: list[LLMResource] = Field(default_factory=list)


class LLMExtraction(BaseModel):
    experiences: list[LLMExperience] = Field(default_factory=list)


# --------------------------------------------------------------------------- stored records
class TopicMention(BaseModel):
    original: str                      # wording used in the source
    canonical: str                     # normalised name
    category: str = "Other"


class Question(BaseModel):
    text: str
    kind: str = "other"
    page: Optional[int] = None


class RoundInfo(BaseModel):
    name: str
    category: str = "Other"
    description: Optional[str] = None
    topics: list[TopicMention] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    difficulty: Optional[str] = None   # only if explicitly stated in the source
    pages: list[int] = Field(default_factory=list)


class Tip(BaseModel):
    text: str
    category: str = "Interview strategy"
    page: Optional[int] = None


class Resource(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    topic: Optional[str] = None
    page: Optional[int] = None


class ExperienceRecord(BaseModel):
    experience_id: str
    company: str                       # folder name, from the directory hierarchy
    company_id: str
    year: str                          # from the directory hierarchy
    role: Optional[str] = None
    rounds: list[RoundInfo] = Field(default_factory=list)
    tips: list[Tip] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    filename: str
    source_path: str
    pages: list[int] = Field(default_factory=list)
    content_sha256: str = ""
