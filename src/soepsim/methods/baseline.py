"""
Non-LLM baseline.

Both focal questions are asked only in wave T, so there is no previous-wave
answer to fall back on. The reference implementation faces the same situation
for ATP and uses uniformly random answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_baseline(human_df: pd.DataFrame, letters: list[str], seed: int = 0) -> pd.DataFrame:
    """
    Draw a uniformly random option for each respondent. Read
    this baseline as a floor on *informativeness*, not as a competitor.
    """
    rng = np.random.default_rng(seed)
    out = human_df[["id"]].copy()
    out["answer"] = rng.choice(letters, size=len(out))
    return out
