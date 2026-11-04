"""
Persona synthesis (auxiliary step of persona-guided prompting).

Writes a single paragraph, which is appended to the system prompt at generation time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..prompts import build_persona_prompt
from .simulate_hf import get_hf_responses

log = logging.getLogger(__name__)


def generate_personas(
    persona_input: pd.DataFrame,
    cfg: dict,
    language: str = "de",
) -> tuple[pd.DataFrame, dict]:
    """
    Write one persona paragraph per respondent.

    :param persona_input: frame with ``id``, ``demographic_profile``,
        ``history_text`` (produced by ``build_dataset``)
    :param cfg: generation config for the auxiliary model
    :return: ``(frame with an added 'persona_description' column, metadata)``
    """
    df = persona_input.copy()
    has_history = df["history_text"].fillna("").str.strip().ne("")
    log.info("Generating personas for %d respondents (%d without history text)",
             len(df), int((~has_history).sum()))

    prompts = [
        build_persona_prompt(row.demographic_profile, row.history_text or "", language)
        for row in df.itertuples()
    ]

    # Personas are prose, not a single letter, so they need real headroom.
    cfg = {**cfg, "max_new_tokens": cfg.get("max_new_tokens", 400)}
    texts, meta = get_hf_responses(prompts, cfg)

    df["persona_description"] = [t.strip() for t in texts]
    empty = int(df["persona_description"].str.len().eq(0).sum())
    if empty:
        log.warning("%d persona generations came back empty.", empty)

    meta["n_personas"] = len(df)
    meta["n_empty_personas"] = empty
    meta["mean_persona_chars"] = float(df["persona_description"].str.len().mean())
    return df, meta


def load_or_generate(
    persona_path: Path, persona_input: pd.DataFrame, cfg: dict, language: str = "de"
) -> pd.DataFrame:
    """Reuse personas from disk when present."""
    if persona_path.exists():
        log.info("Reusing personas from %s", persona_path)
        return pd.read_csv(persona_path)
    df, meta = generate_personas(persona_input, cfg, language)
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(persona_path, index=False)
    log.info("Wrote %d personas to %s (%s)", len(df), persona_path, meta)
    return df
