"""Lightweight lexical matching used to ground extracted items and verify citations.

This is a *heuristic* (content-word overlap on 5-character stems), not entailment.
It reliably catches invented content; it can miss subtle misreadings.
"""
from __future__ import annotations

import re

_STOP = set(
    """a an the and or but if then else for of to in on at by with from as is are was were be been being
    it its this that these those there here they them their he she his her we our you your i me my
    not no yes do does did done have has had having can could should would will shall may might must
    about into over under than so such also very just more most some any all each other another which who whom
    what when where why how while during after before between across per via using use used etc eg ie
    reported mentioned mention mentions candidate candidates student students experience experiences
    according document documents passage passages source sources inference answer asked ask""".split()
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\+#]*")


def content_words(text: str) -> set[str]:
    """Distinctive lowercase word stems (first 5 chars) of ``text``."""
    words = set()
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP or (len(tok) < 2 and tok not in {"c"}):
            continue
        words.add(tok[:5])
    return words


def support_ratio(claim: str, evidence: str) -> float:
    """Fraction of the claim's content words that appear in the evidence text."""
    cw = content_words(claim)
    if not cw:
        return 0.0
    return len(cw & content_words(evidence)) / len(cw)
