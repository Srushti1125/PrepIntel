"""Topic normalisation: deterministic alias -> canonical name + dashboard category."""
from __future__ import annotations

import re

ROUND_CATEGORIES = [
    "Online Assessment",
    "Technical Interview",
    "Managerial Interview",
    "HR Interview",
    "Group Discussion",
    "Other",
]
TIP_CATEGORIES = [
    "OA preparation",
    "Technical preparation",
    "Coding preparation",
    "Project preparation",
    "HR preparation",
    "Interview strategy",
]
QUESTION_KINDS = ["coding", "technical", "project", "hr", "aptitude", "other"]
TOPIC_CATEGORIES = [
    "DSA and coding",
    "Programming languages",
    "OOPs",
    "DBMS and SQL",
    "Operating systems",
    "Computer networks",
    "Aptitude",
    "Projects",
    "System design",
    "HR and behavioral",
    "Other",
]

# canonical name -> (category, [aliases])
_CANON: dict[str, tuple[str, list[str]]] = {
    "Data Structures and Algorithms": ("DSA and coding", ["dsa", "data structures", "algorithms", "ds and algo", "ds algo", "data structure"]),
    "Arrays": ("DSA and coding", ["array"]),
    "Strings": ("DSA and coding", ["string"]),
    "Linked Lists": ("DSA and coding", ["linked list", "ll"]),
    "Stacks and Queues": ("DSA and coding", ["stack", "stacks", "queue", "queues"]),
    "Trees": ("DSA and coding", ["tree", "binary tree", "bst", "binary search tree"]),
    "Graphs": ("DSA and coding", ["graph", "bfs", "dfs"]),
    "Dynamic Programming": ("DSA and coding", ["dp"]),
    "Recursion and Backtracking": ("DSA and coding", ["recursion", "backtracking"]),
    "Greedy Algorithms": ("DSA and coding", ["greedy"]),
    "Sorting and Searching": ("DSA and coding", ["sorting", "searching", "binary search"]),
    "Hashing": ("DSA and coding", ["hash map", "hashmap", "hash table", "hash"]),
    "Heaps": ("DSA and coding", ["heap", "priority queue"]),
    "C++": ("Programming languages", ["cpp", "c plus plus"]),
    "C": ("Programming languages", []),
    "Java": ("Programming languages", []),
    "Python": ("Programming languages", []),
    "JavaScript": ("Programming languages", ["js"]),
    "Object-Oriented Programming": ("OOPs", ["oop", "oops", "object oriented programming", "object oriented", "inheritance", "polymorphism", "encapsulation", "abstraction"]),
    "Database Management Systems": ("DBMS and SQL", ["dbms", "database", "databases", "normalization", "normalisation", "acid", "transactions", "indexing", "nosql"]),
    "SQL": ("DBMS and SQL", ["sql queries", "joins", "sql joins"]),
    "Operating Systems": ("Operating systems", ["os", "operating system", "process scheduling", "deadlock", "deadlocks", "paging", "memory management", "multithreading", "threads", "synchronization", "semaphores"]),
    "Computer Networks": ("Computer networks", ["cn", "networking", "networks", "computer network", "tcp ip", "tcp/ip", "osi model", "osi", "http", "dns"]),
    "Quantitative Aptitude": ("Aptitude", ["quant", "quantitative", "aptitude", "maths", "math"]),
    "Logical Reasoning": ("Aptitude", ["reasoning", "logical", "puzzles", "puzzle"]),
    "Verbal Ability": ("Aptitude", ["verbal", "english", "comprehension"]),
    "Projects": ("Projects", ["project", "project discussion", "project questions", "resume projects"]),
    "System Design": ("System design", ["hld", "lld", "high level design", "low level design", "design"]),
    "HR and Behavioral Questions": ("HR and behavioral", ["hr", "behavioral", "behavioural", "hr questions", "tell me about yourself", "strengths and weaknesses", "situational"]),
}

_ALIAS: dict[str, str] = {}
for _canon, (_cat, _aliases) in _CANON.items():
    _ALIAS[_canon.lower()] = _canon
    for _a in _aliases:
        _ALIAS[_a.lower()] = _canon

# Alias search inside longer phrases: only aliases of >=3 chars, longest first.
_FUZZY = sorted((a for a in _ALIAS if len(a) >= 3 and a.isascii()), key=len, reverse=True)


def _clean(text: str) -> str:
    text = re.sub(r"[^a-z0-9+#/ ]+", " ", text.lower().replace("-", " "))
    return re.sub(r"\s+", " ", text).strip()


def normalize_topic(original: str) -> tuple[str, str]:
    """Return ``(canonical_name, category)``. Unknown topics keep their own (title-cased) name."""
    key = _clean(original)
    if not key:
        return original.strip(), "Other"
    canon = _ALIAS.get(key)
    if canon is None:
        padded = f" {key} "
        for alias in _FUZZY:
            if f" {_clean(alias)} " in padded:
                canon = _ALIAS[alias]
                break
    if canon is None:
        return original.strip().title() if original.islower() else original.strip(), "Other"
    return canon, _CANON[canon][0]


def coerce(value: str | None, allowed: list[str], default: str) -> str:
    """Map a free-text label onto one of ``allowed`` (case-insensitive), else ``default``."""
    if not value:
        return default
    v = value.strip().lower()
    for a in allowed:
        if a.lower() == v:
            return a
    for a in allowed:  # loose containment e.g. "technical" -> "Technical Interview"
        if v in a.lower() or a.lower() in v:
            return a
    return default
