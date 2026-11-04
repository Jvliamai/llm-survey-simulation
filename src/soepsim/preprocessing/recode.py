"""Recoding utilities: missing conventions, derived variables, category collapsing."""

from __future__ import annotations
import numpy as np
import pandas as pd
from .variables import FocalQuestion


def decode_missings(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Replace SOEP negative missing codes (-1 .. -9) with ``NaN``.

    :param df: input frame (not modified in place)
    :param columns: restrict to these columns; default = all numeric columns
    :return: copy with missings decoded
    """
    out = df.copy()
    cols = columns if columns is not None else out.select_dtypes("number").columns.tolist()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out.loc[out[c] < 0, c] = np.nan
    return out


def missing_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column count and share of missings, for the preprocessing appendix."""
    rows = []
    for c in columns:
        if c not in df.columns:
            rows.append({"variable": c, "present": False, "n_missing": np.nan, "share": np.nan})
            continue
        n_miss = int(df[c].isna().sum())
        rows.append({
            "variable": c,
            "present": True,
            "n_missing": n_miss,
            "share": round(n_miss / max(len(df), 1), 4),
        })
    return pd.DataFrame(rows)


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age at the time of interview, approximated from birth year and survey year
    (``syear - gebjahr``). 
    """
    out = df.copy()
    age = out["syear"] - out["gebjahr"]

    if "d11101" in out.columns:
        age = age.fillna(out["d11101"])
    out["age"] = age.where((age >= 16) & (age <= 110))
    return out


def add_isco_major(df: pd.DataFrame) -> pd.DataFrame:
    """First digit of the 4-digit ISCO-08 code (major occupational group)."""
    out = df.copy()
    if "pgisco08" not in out.columns:
        out["pgisco08_major"] = np.nan
        return out
    codes = pd.to_numeric(out["pgisco08"], errors="coerce")
    # ISCO-08 is 4 digits; major group = leading digit. Guard against 3-digit codes.
    major = np.where(codes >= 1000, codes // 1000, np.where(codes >= 100, codes // 100, codes))
    out["pgisco08_major"] = pd.Series(major, index=out.index).where(codes.notna())
    return out


_NACE_SECTION_RANGES = (
    (1, 3, 1),    # A - Land-/Forstwirtschaft, Fischerei
    (5, 9, 2),    # B - Bergbau
    (10, 33, 3),  # C - Verarbeitendes Gewerbe
    (35, 35, 4),  # D - Energieversorgung
    (36, 39, 5),  # E - Wasser/Abfall
    (41, 43, 6),  # F - Baugewerbe
    (45, 47, 7),  # G - Handel
    (49, 53, 8),  # H - Verkehr und Lagerei
    (55, 56, 9),  # I - Gastgewerbe
    (58, 63, 10), # J - Information und Kommunikation
    (64, 66, 11), # K - Finanz-/Versicherungsdienstleistungen
    (68, 68, 12), # L - Grundstuecks- und Wohnungswesen
    (69, 75, 13), # M - Freiberufliche/wiss./techn. Dienstleistungen
    (77, 82, 14), # N - Sonstige wirtschaftliche Dienstleistungen
    (84, 84, 15), # O - Oeffentliche Verwaltung
    (85, 85, 16), # P - Erziehung und Unterricht
    (86, 88, 17), # Q - Gesundheits-/Sozialwesen
    (90, 93, 18), # R - Kunst, Unterhaltung, Erholung
    (94, 96, 19), # S - Sonstige Dienstleistungen
    (97, 98, 20), # T - Private Haushalte
    (99, 99, 21), # U - Exterritoriale Organisationen
)


def add_nace_section(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pgnace2" not in out.columns:
        out["pgnace_section"] = np.nan
        return out
    codes = pd.to_numeric(out["pgnace2"], errors="coerce")
    conditions = [codes.between(lo, hi) for lo, hi, _ in _NACE_SECTION_RANGES]
    choices = [sec for _, _, sec in _NACE_SECTION_RANGES]
    out["pgnace_section"] = np.select(conditions, choices, default=np.nan)
    out.loc[codes.isna(), "pgnace_section"] = np.nan
    return out


_PGSTIB_GROUPS: list[tuple[float, float, int]] = [
    (10, 15, 1),
    (110, 130, 1),
    (210, 250, 2),
    (410, 440, 3),
    (510, 540, 4),
    (610, 640, 5),
]

PGSTIB_GROUP_LABELS = {
    1: "nicht erwerbstaetig / in Ausbildung",
    2: "Arbeiter(in)",
    3: "Selbstaendige(r)",
    4: "Angestellte(r)",
    5: "Beamte(r)",
}


def add_pgstib_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pgstib" not in out.columns:
        out["pgstib_group"] = np.nan
        return out
    codes = pd.to_numeric(out["pgstib"], errors="coerce")
    conditions = [(codes >= lo) & (codes <= hi) for lo, hi, _ in _PGSTIB_GROUPS]
    choices = [group for _, _, group in _PGSTIB_GROUPS]
    out["pgstib_group"] = pd.Series(
        np.select(conditions, choices, default=np.nan), index=out.index
    ).where(codes.notna())
    return out


def oecd_equivalence_factor(hh_size: pd.Series, n_children: pd.Series) -> pd.Series:
    """
    Modified OECD equivalence scale: 1.0 for the first adult, 0.5 per further
    adult, 0.3 per child.
    """
    size = pd.to_numeric(hh_size, errors="coerce")
    kids = pd.to_numeric(n_children, errors="coerce").fillna(0)
    adults = (size - kids).clip(lower=1)
    return 1.0 + 0.5 * (adults - 1) + 0.3 * kids


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    lo, hi = numeric.quantile([lower, upper])
    return numeric.clip(lower=lo, upper=hi)


def add_equivalised_income(df: pd.DataFrame) -> pd.DataFrame:
    """Equivalised household income and its quintile (within the target wave)."""
    out = df.copy()
    factor = oecd_equivalence_factor(out.get("d11106"), out.get("d11107"))
    out["equiv_income"] = pd.to_numeric(out.get("i11102"), errors="coerce") / factor

    valid = out["equiv_income"].notna()
    out["income_quintile"] = np.nan
    if valid.sum() >= 5:
        out.loc[valid, "income_quintile"] = (
            pd.qcut(out.loc[valid, "equiv_income"], 5, labels=[1, 2, 3, 4, 5])
            .astype(float)
        )
    return out


def add_subgroup_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coarse subgroup variables used for subgroup bias reporting (Extension A)
    and stratified rectification (Extension B).
    """
    out = df.copy()

    # Education, 3 levels from CASMIN
    cas = pd.to_numeric(out.get("pgcasmin"), errors="coerce")
    out["edu3"] = np.select(
        [cas.isin([0, 1, 2, 3]), cas.isin([4, 5]), cas.isin([6, 7, 8, 9])],
        [1, 2, 3],
        default=np.nan,
    )

    # Age, 3 levels
    age = pd.to_numeric(out.get("age"), errors="coerce")
    out["age3"] = np.select(
        [age < 35, (age >= 35) & (age < 55), age >= 55],
        [1, 2, 3],
        default=np.nan,
    )
    return out


EDU3_LABELS = {1: "niedrig", 2: "mittel", 3: "hoch"}
AGE3_LABELS = {1: "18-34", 2: "35-54", 3: "55+"}


#: Bundestag-relevant parties kept as their own category in plh0012_h.
_PARTY_KEEP = {1: "SPD", 2: "CDU", 3: "CSU", 4: "FDP",
               5: "Buendnis 90/Die Gruenen", 6: "Die Linke", 27: "AfD"}


def collapse_party(series: pd.Series) -> pd.Series:
    """
    Collapse the 32-category party variable to the seven Bundestag-relevant
    parties plus 'andere Partei'.
    """
    codes = pd.to_numeric(series, errors="coerce")
    out = codes.map(_PARTY_KEEP)
    out = out.where(codes.isna() | codes.isin(_PARTY_KEEP), "andere Partei")
    return out


def collapse_nationality(series: pd.Series) -> pd.Series:
    """Collapse pgnation to deutsch / nicht-deutsch."""
    codes = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out[codes == 1] = "deutsch"
    out[codes.notna() & (codes != 1)] = "nicht-deutsch"
    return out


def collapse_trust_to_four(series: pd.Series) -> pd.Series:
    """
    Collapse the 0-10 trust scale of focal question 2 onto four levels, so the
    answer format matches ATP Q2 (four options, one letter).

    Cut points 0-2 / 3-5 / 6-8 / 9-10 map to 1..4 with 4 = highest trust,
    preserving the 'higher is better' orientation of the reference
    implementation. The cut points are a modelling choice; this variant is
    reported as a sensitivity analysis, not as the headline result.
    """
    v = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [v <= 2, (v >= 3) & (v <= 5), (v >= 6) & (v <= 8), v >= 9],
            [1, 2, 3, 4],
            default=np.nan,
        ),
        index=series.index,
    )


# Focal questions
def encode_focal(df: pd.DataFrame, fq: FocalQuestion) -> pd.DataFrame:
    """
    Turn the raw focal-question column into the prompt letter used as the
    ground-truth label.

    Adds two columns:
      ``correct_answer``  prompt letter (A, B, C, ...) or NaN
      ``focal_raw``       the raw SOEP code, kept for auditing
    """
    out = df.copy()
    raw = pd.to_numeric(out[fq.variable], errors="coerce")
    out["focal_raw"] = raw
    out["correct_answer"] = raw.map(fq.soep_code_to_letter)
    return out


def phi(letters: pd.Series | list[str], fq: FocalQuestion) -> np.ndarray:
    """
    The mapping phi: Y -> R.

    Deterministic by construction, as required by the rectification step: any
    randomness here would inflate the variance of the correction term.
    """
    if isinstance(letters, pd.Series):
        return letters.map(fq.response_map).astype(float).values
    return np.array([fq.response_map.get(x, np.nan) for x in letters], dtype=float)


def carry_forward(
    panel: pd.DataFrame, columns: list[str], id_col: str = "pid", time_col: str = "syear"
) -> pd.DataFrame:
    """
    Fill gaps in rotation-module variables with the person's most recent earlier
    valid answer.
    """
    out = panel.sort_values([id_col, time_col]).copy()
    present = [c for c in columns if c in out.columns]
    if present:
        out[present] = out.groupby(id_col)[present].ffill()
    return out
