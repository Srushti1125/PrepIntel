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


_DISPLAY_TOPIC_ALIASES = {
    "aiml": "AI/ML", "ai ml": "AI/ML", "ai/ml": "AI/ML", "ml": "Machine Learning",
    "genai": "Generative AI", "gen ai": "Generative AI", "generative ai": "Generative AI",
    "numpy": "NumPy", "powerbi": "Power BI", "postgresql": "PostgreSQL",
    "hashmaps": "Hashing", "hash maps": "Hashing",
    "group by": "GROUP BY", "order by": "ORDER BY", "join": "JOIN", "limit": "LIMIT",
    "logistic regression": "Logistic Regression", "problem solving": "Problem Solving",
    "date functions": "Date Functions", "data interpretation": "Data Interpretation",
    "data analytics": "Data Analytics", "file handling": "File Handling", "linux commands": "Linux Commands",
    "hackathon experience": "Hackathon Experience", "core cs fundamentals": "Core CS Fundamentals",
    "equals method": ".Equals() Method", "rag retrieval augmented generation": "RAG (Retrieval-Augmented Generation)",
    "binary trees/bsts": "Binary Trees/BSTs", "mysql": "MySQL", "api": "API", "apis": "API",
    "llm": "LLM", "llms": "LLM", "react js": "React", "jwt tokens": "JWT Authentication",
    "jwt authentication": "JWT Authentication", "jwt auth": "JWT Authentication",
    "simple interest compound interest": "Simple and Compound Interest",
    "simple and compound interest": "Simple and Compound Interest",
}


def display_topic_name(value: str) -> str:
    """Return a consistent display label for known spelling and punctuation variants."""
    key = _clean(value)
    return _DISPLAY_TOPIC_ALIASES.get(key, _ALIAS.get(key, value.strip()))


def canonical_resource_topics(value: str | None) -> list[str]:
    """Map free-form or multi-topic resource labels to consistent topic groups."""
    text = (value or "").strip()
    if not text or _clean(text) in {"general", "general resources"}:
        return ["General resources"]
    rules = (
        (r"\b(?:data structures and algorithms|dsa)\b", "Data Structures and Algorithms"),
        (r"\b(?:competitive programming)\b", "Competitive Programming"),
        (r"\b(?:coding practice|coding problems?|coding questions?|coding)\b", "Coding"),
        (r"\b(?:aptitude|quantitative|quants?)\b", "Aptitude"),
        (r"\b(?:puzzles?|logical reasoning|logic building)\b", "Logical Reasoning"),
        (r"\b(?:guesstimates?)\b", "Guesstimates"),
        (r"\b(?:probability)\b", "Probability"),
        (r"\b(?:mathematics|maths?)\b", "Mathematics"),
        (r"\b(?:sql)\b", "SQL"),
        (r"\b(?:dbms|database management)\b", "Database Management Systems"),
        (r"\b(?:operating systems?|os)\b", "Operating Systems"),
        (r"\b(?:computer networks?|ccn|cn)\b", "Computer Networks"),
        (r"\b(?:oops?|oop|object oriented(?: programming)?)\b", "OOPs"),
        (r"\b(?:java)\b", "Java"),
        (r"\b(?:javascript|js)\b", "JavaScript"),
        (r"(?<![a-z0-9])(?:c\+\+|cpp)(?![a-z0-9])", "C++"),
        (r"\b(?:objective c|objective-c)\b", "Objective-C"),
        (r"\b(?:python)\b", "Python"),
        (r"\b(?:react(?:\.js|js)?)\b", "React"),
        (r"\b(?:system design|low level design|high level design|lld|hld)\b", "System Design"),
        (r"\b(?:oa|online assessment|assessment|mcqs?)\b", "Assessment and MCQs"),
        (r"\b(?:machine learning|deep learning|\bml\b|ai/ml|aiml|genai|generative ai|ai tools?)\b", "AI/ML and Data Science"),
        (r"\b(?:finance|investment banking)\b", "Finance"),
        (r"\b(?:case studies|case study)\b", "Case Studies"),
        (r"\b(?:hr|hr questions?|human resources|behavioral)\b", "HR and Behavioral"),
        (r"\b(?:company research|company values|company fit)\b", "Company Research"),
        (r"\b(?:resume|ats)\b", "Resume and ATS"),
        (r"\b(?:product management|product manager)\b", "Product Management"),
        (r"\b(?:core subjects?|core cs|cs fundamentals|computer fundamentals)\b", "Core CS Fundamentals"),
        (r"\b(?:strings?|string generation)\b", "Strings"),
        (r"\b(?:problem solving)\b", "Problem Solving"),
        (r"\b(?:interview prep|interview preparation)\b", "Interview Preparation"),
        (r"\b(?:research)\b", "Research"),
        (r"\b(?:data analytics|data analysis)\b", "Data Analytics"),
    )
    found = [label for pattern, label in rules if re.search(pattern, text, re.I)]
    return list(dict.fromkeys(found)) or [display_topic_name(text)]


def normalize_topic(original: str) -> tuple[str, str]:
    """Return ``(canonical_name, category)``. Unknown topics keep their own (title-cased) name."""
    key = _clean(original)
    if not key:
        return original.strip(), "Other"
    display_alias = _DISPLAY_TOPIC_ALIASES.get(key)
    if display_alias:
        return display_alias, _CANON.get(display_alias, ("Other", []))[0]
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
