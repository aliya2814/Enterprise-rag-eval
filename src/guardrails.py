"""Guardrails: refuse when context is weak, and screen obvious prompt injection.

These are deliberately simple and transparent - the README explains the trade-offs.
A senior portfolio shows you THOUGHT about safety, not that you used a heavy lib.
"""
from __future__ import annotations
import re

from .config import settings

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)?(previous|above|prior)?\s*instructions",
    r"disregard (your|the) (system|previous) prompt",
    r"reveal (your|the) (system )?prompt",
    r"you are now",
    r"act as (a )?(dan|developer mode)",
]


def is_injection(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _INJECTION_PATTERNS)


def context_is_weak(hits) -> bool:
    """If the best retrieved chunk scores below threshold, we lack support."""
    if not hits:
        return True
    return max(h.score for h in hits) < settings.min_context_score


def refusal_message(reason: str) -> str:
    return (f"I can't answer that confidently. Reason: {reason}. "
            "Try rephrasing, or ask about content covered in the filings.")
