"""Column-selective loading of SOEP-Core long-format files."""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from .variables import ALL_YEARS, READ_PLAN

log = logging.getLogger(__name__)

_EXTENSIONS = (".dta", ".sav", ".csv", ".csv.gz", ".parquet")


def find_soep_file(soep_dir: Path, name: str) -> Path:
    """
    Locate a SOEP dataset file by stem, trying the usual extensions.
    """
    candidates = [soep_dir, soep_dir / "soepdata"]
    for base in candidates:
        for ext in _EXTENSIONS:
            p = base / f"{name}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Could not find SOEP dataset {name!r} under {soep_dir} "
        f"(tried {', '.join(_EXTENSIONS)} in {[str(c) for c in candidates]}). "
        "Point --soep-dir at the directory that contains ppathl/pl/pgen/..."
    )


def _read_any(path: Path, columns: list[str]) -> pd.DataFrame:
    suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".csv.gz") else path.suffix

    if suffix == ".dta":
        import pyreadstat

        df, _ = pyreadstat.read_dta(str(path), usecols=columns)
        return df
    if suffix == ".sav":
        import pyreadstat

        df, _ = pyreadstat.read_sav(str(path), usecols=columns)
        return df
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix in (".csv", ".csv.gz"):
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported file type: {path}")


def load_dataset(
    soep_dir: Path, name: str, years: tuple[int, ...] = ALL_YEARS
) -> pd.DataFrame:
    if name not in READ_PLAN:
        raise KeyError(f"No read plan for dataset {name!r}")
    path = find_soep_file(soep_dir, name)

    wanted = list(READ_PLAN[name])
    available = _available_columns(path)
    missing = [c for c in wanted if c not in available]
    if missing:
        log.warning(
            "%s: %d planned column(s) not in the delivered file and skipped: %s",
            name, len(missing), ", ".join(missing),
        )
    usecols = [c for c in wanted if c in available]

    df = _read_any(path, usecols)
    if "syear" in df.columns:
        df = df[df["syear"].isin(years)]
    log.info("Loaded %s: %d rows x %d cols from %s", name, len(df), df.shape[1], path.name)
    return df.reset_index(drop=True)


def _available_columns(path: Path) -> set[str]:
    suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".csv.gz") else path.suffix
    if suffix in (".dta", ".sav"):
        import pyreadstat

        reader = pyreadstat.read_dta if suffix == ".dta" else pyreadstat.read_sav
        _, meta = reader(str(path), metadataonly=True)
        return set(meta.column_names)
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    return set(pd.read_csv(path, nrows=0).columns)


def load_panel(soep_dir: Path, years: tuple[int, ...] = ALL_YEARS) -> pd.DataFrame:
    """
    Build the person-year panel by merging all planned SOEP files.

    :return: person-year frame keyed by (pid, syear)
    """
    ppathl = load_dataset(soep_dir, "ppathl", years)
    panel = ppathl

    for name in ("pgen", "pequiv", "pl"):
        df = load_dataset(soep_dir, name, years)
        df = df.drop(columns=[c for c in ("hid",) if c in df.columns])
        panel = panel.merge(df, on=["pid", "syear"], how="left", validate="1:1")

    for name in ("hgen", "hbrutto"):
        df = load_dataset(soep_dir, name, years)
        panel = panel.merge(df, on=["hid", "syear"], how="left", validate="m:1")

    log.info("Panel: %d person-years, %d unique persons, waves %s",
             len(panel), panel["pid"].nunique(), sorted(panel["syear"].unique()))
    return panel
