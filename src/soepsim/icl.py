"""In-context learning as the prompting counterpart of Domain-FT."""
 
from __future__ import annotations
 
import logging
 
import numpy as np
import pandas as pd
 
log = logging.getLogger(__name__)
 
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
STRATEGIES = ("similar", "random")
 
_EMBEDDER_CACHE: dict = {}
 
 
def _embed(texts: list[str], model_name: str) -> np.ndarray:
    """L2-normalised sentence embeddings (lazy import, cached model)."""
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer
 
        log.info("Loading embedding model %s", model_name)
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    emb = _EMBEDDER_CACHE[model_name].encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=128
    )
    return np.asarray(emb, dtype="float32")
 
 
def check_disjoint(pool: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Fail loudly if a pool respondent is also a target (would leak labels)."""
    for col in ("id", "pid"):
        if col in pool.columns and col in targets.columns:
            overlap = set(pool[col]) & set(targets[col])
            if overlap:
                raise ValueError(
                    f"ICL pool and evaluation frame share {len(overlap)} '{col}' values. "
                    "Pool labels would leak into the correction set -- rebuild the split."
                )
 
 
def select_icl_examples(
    pool: pd.DataFrame,
    targets: pd.DataFrame,
    k: int,
    strategy: str = "similar",
    seed: int = 0,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> dict:
    """
    Select k labelled examples from ``pool`` for each target respondent.

    :param pool: columns ``id``, ``demographic_profile``, ``answer``
    :param targets: columns ``id``, ``demographic_profile``
    :param k: number of examples (at most the pool size)
    :return: ``{target_id: [(profile, answer), ...]}``
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    check_disjoint(pool, targets)
    pool = pool.dropna(subset=["demographic_profile", "answer"]).reset_index(drop=True)
    if pool.empty or k <= 0:
        return {tid: [] for tid in targets["id"]}
    if k > len(pool):
        log.warning("icl_k=%d exceeds the pool size %d -- using k=%d", k, len(pool), len(pool))
        k = len(pool)
 
    pairs = list(zip(pool["demographic_profile"], pool["answer"]))
 
    if strategy == "random":
        rng = np.random.default_rng(seed)
        chosen = [pairs[i] for i in rng.choice(len(pool), size=k, replace=False)]
        return {tid: chosen for tid in targets["id"]}
 
    # strategy == "similar": cosine similarity of profile embeddings
    pool_emb = _embed(pool["demographic_profile"].fillna("").tolist(), embed_model)
    tgt_emb = _embed(targets["demographic_profile"].fillna("").tolist(), embed_model)
    out: dict = {}
    for start in range(0, len(targets), 1024):
        sims = tgt_emb[start:start + 1024] @ pool_emb.T
        top = np.argsort(-sims, axis=1)[:, :k]
        for tid, idx in zip(targets["id"].iloc[start:start + 1024], top):
            # most similar example last, i.e. closest to the target's own question
            out[tid] = [pairs[i] for i in idx[::-1]]
    return out