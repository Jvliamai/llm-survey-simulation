"""Rendering of demographic profiles and response histories as German text. """

from __future__ import annotations

import numpy as np
import pandas as pd

from .variables import (
    COVARIATES,
    HistoryItem,
)

MISSING_TEXT = "keine Angabe"


def _render_value(cov, value) -> str:
    """Render one covariate value as German text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING_TEXT
    if isinstance(value, str):
        return value if value.strip() else MISSING_TEXT

    if cov.kind == "numeric":
        if cov.formatter is not None:
            try:
                return cov.formatter(float(value))
            except (TypeError, ValueError):
                return MISSING_TEXT
        return f"{value:g}"

    # categorical
    try:
        code = int(round(float(value)))
    except (TypeError, ValueError):
        return MISSING_TEXT
    if cov.value_labels:
        return cov.value_labels.get(code, MISSING_TEXT)
    return str(code)


def render_profile(row: pd.Series, covariates=COVARIATES) -> str:
    """
    Render one respondent's demographic frame X_i as a single string.

    :param row: a row of the analysis frame
    :param covariates: covariate registry (defaults to the full 25-variable set)
    :return: ``"Alter: 42 Jahre; Geschlecht: weiblich; ..."``
    """
    parts = []
    for cov in covariates:
        # collapsed string columns take precedence over the raw numeric ones
        collapsed = f"{cov.name}_label"
        value = row[collapsed] if collapsed in row.index else row.get(cov.name)
        parts.append(f"{cov.label_de}: {_render_value(cov, value)}")
    return "; ".join(parts)


def _render_history_value(item: HistoryItem, value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        return value or None
    try:
        code = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if item.value_labels:
        return item.value_labels.get(code)
    return str(code)


def render_history(
    person_waves: pd.DataFrame,
    items: list[HistoryItem],
    year_col: str = "syear",
) -> str:
    """
    Render one respondent's answers across the history waves.
    Waves in which the person was not interviewed are skipped entirely.
    """
    blocks: list[str] = []
    for year, wave in person_waves.sort_values(year_col).groupby(year_col):
        rec = wave.iloc[0]
        lines = []
        for item in items:
            if item.name not in rec.index:
                continue
            text = _render_history_value(item, rec[item.name])
            if text is None:
                continue
            suffix = f" ({item.scale_hint})" if item.scale_hint and not item.value_labels else ""
            lines.append(f"- {item.question_de}{suffix}: {text}")
        if lines:
            blocks.append(f"Befragung {int(year)}:\n" + "\n".join(lines))
    return "\n\n".join(blocks)


# Diagnostic
def history_coverage(person_waves: pd.DataFrame, items: list[HistoryItem]) -> dict:
    n_waves = int(person_waves["syear"].nunique())
    names = [i.name for i in items if i.name in person_waves.columns]
    n_answers = int(person_waves[names].notna().sum().sum()) if names else 0
    return {
        "n_history_waves": n_waves,
        "n_history_answers": n_answers,
        "n_history_items_possible": len(names) * n_waves,
    }
