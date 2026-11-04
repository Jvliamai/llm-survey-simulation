"""Shared helpers: answer parsing, file discovery, method naming."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path


# Method naming
METHOD_LABELS: dict[str, str] = {
    "silicon_sampling": "Demo-only",
    "concept_guided": "Persona-guided",
    "ft": "Domain-FT",
    "icl": "In-context learning",   # Extension C
    "random": "Baseline (random)",
}

METHODS_ORDER: list[str] = ["concept_guided", "ft", "icl", "silicon_sampling", "random"]

MODEL_LABELS: dict[str, str] = {
    "llama": "Llama-3.1-8B-Instruct",
    "ministral": "Ministral-8B-Instruct-2410",
    "qwen": "Qwen3-8B",
}

BASELINE_METHODS = {"random"}


def extract_method(model_name: str) -> str:
    for m in METHODS_ORDER:
        if f":{m}" in model_name:
            return m
    return "other"


def find_model_csvs(base_dir: str) -> list[str]:
    """All prediction CSVs under ``base_dir``, excluding ``_old`` directories."""
    return [
        p
        for p in sorted(glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True))
        if "_old" not in Path(p).parts
    ]


# Multiple-choice parsing
def get_multi_choice_info(choices: list[str] | tuple[str, ...]) -> tuple[dict, list[str]]:
    """
    Build the lookup structures the parser needs.

    :param choices: answer options in prompt order
    :return: ``({"A": "Viel hoeher ...", ...}, ["A", "B", ...])``
    """
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    index2ans = {letter: str(text) for letter, text in zip(letters, choices)}
    return index2ans, letters


def parse_multi_choice_response(
    response: str, all_choices: list[str], index2ans: dict[str, str]
) -> str | None:
    """
    Extract the chosen option letter from a raw model completion. Imported by the reference implementation.

    :return: the option letter, or ``None`` if nothing could be extracted
    """
    if response is None:
        return None
    text = str(response).strip()
    if not text:
        return None

    upper = text.upper()

    # exact single letter
    if upper in all_choices:
        return upper

    # delimited letter
    for choice in all_choices:
        patterns = [
            rf"(?<![A-Z]){re.escape(choice)}(?![A-Z])",
        ]
        for pat in patterns:
            m = re.search(pat, upper)
            if m:
                # reject matches inside a longer word, e.g. "A" in "ANTWORT"
                start, end = m.span()
                before = upper[start - 1] if start > 0 else " "
                after = upper[end] if end < len(upper) else " "
                if not before.isalpha() and not after.isalpha():
                    return choice

    # a purely numeric answer is an option index (0-based).
    m = re.fullmatch(r"[^\d]*?(\d{1,2})[^\d]*", text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(all_choices):
            return all_choices[idx]
        return None

    # verbatim option text
    hits = [
        letter
        for letter, ans in index2ans.items()
        if ans and ans.strip() and ans.strip().lower() in text.lower()
    ]
    if len(hits) == 1:
        return hits[0]

    return None


def parse_rate(parsed: list[str | None]) -> float:
    """Share of generations that could be parsed. Reported as a validity check."""
    if not parsed:
        return float("nan")
    return sum(p is not None for p in parsed) / len(parsed)
