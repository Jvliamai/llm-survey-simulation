"""Build the analysis dataset for one focal question."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from ..prompts import build_ft_example, build_prompt
from . import recode
from .profiles import render_history, render_profile
from .soep_loader import load_panel
from .variables import (
    COVARIATES,
    FOCAL_QUESTIONS,
    HISTORY_YEARS,
    NETTO_INTERVIEW_RANGE,
    NON_REPRESENTATIVE_PSAMPLES,
    SUBGROUP_VARS,
    TARGET_YEAR,
    history_items,
)

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger(__name__)

ATP_MIRROR_N = {"q1": 691, "q2": 643}


def prepare_panel(soep_dir: Path) -> pd.DataFrame:
    panel = load_panel(soep_dir)

    focal_vars = [fq.variable for fq in FOCAL_QUESTIONS.values()]
    protect = {"pid", "hid", "cid", "syear", "netto", "psample", *focal_vars}
    panel = recode.decode_missings(
        panel, columns=[c for c in panel.columns if c not in protect]
    )
    for c in focal_vars:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
            panel.loc[panel[c] < 0, c] = np.nan

    panel = recode.carry_forward(panel, ["pli0096_h", "pli0098_h", "pgnation", "pgnace2"])
    panel = recode.add_age(panel)
    panel = recode.add_isco_major(panel)
    panel = recode.add_nace_section(panel)
    panel = recode.add_pgstib_group(panel)
    return panel


def select_target_wave(panel: pd.DataFrame, focal_var: str, exclude_special: bool) -> pd.DataFrame:
    df = panel[panel["syear"] == TARGET_YEAR].copy()
    steps = [("wave T", len(df))]

    if "netto" in df.columns:
        lo, hi = NETTO_INTERVIEW_RANGE
        df = df[df["netto"].between(lo, hi)]
        steps.append(("realised interview", len(df)))

    if exclude_special and "psample" in df.columns:
        df = df[~df["psample"].isin(NON_REPRESENTATIVE_PSAMPLES)]
        steps.append(("representative sub-samples", len(df)))

    df = df[df[focal_var].notna()]
    steps.append(("focal question answered", len(df)))

    for label, n in steps:
        log.info("  after %-32s n = %d", label, n)
    df.attrs["flow"] = steps
    return df


def attach_history(
    target: pd.DataFrame, panel: pd.DataFrame, items, require_history: bool
) -> pd.DataFrame:
    hist = panel[panel["syear"].isin(HISTORY_YEARS)]
    hist = hist[hist["pid"].isin(target["pid"])]

    texts, coverage = {}, {}
    for pid, waves in hist.groupby("pid"):
        texts[pid] = render_history(waves, items)
        coverage[pid] = int(waves["syear"].nunique())

    out = target.copy()
    out["history_text"] = out["pid"].map(texts).fillna("")
    out["n_history_waves"] = out["pid"].map(coverage).fillna(0).astype(int)

    if require_history:
        before = len(out)
        out = out[out["n_history_waves"] > 0]
        log.info("  after %-32s n = %d (dropped %d without history)",
                 "history requirement", len(out), before - len(out))
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "plh0012_h" in out.columns:
        out["plh0012_h_label"] = recode.collapse_party(out["plh0012_h"])
    if "pgnation" in out.columns:
        out["pgnation_label"] = recode.collapse_nationality(out["pgnation"])
    return out


def build_prompts(df: pd.DataFrame, fq, language: str) -> pd.DataFrame:
    out = df.copy()
    out["demographic_profile"] = out.apply(render_profile, axis=1)
    out["question"] = fq.question_de
    out["choices"] = [list(fq.choices_de)] * len(out)
    out["prompt"] = [
        json.dumps(
            build_prompt(prof, fq.question_de, fq.choices_de, language=language),
            ensure_ascii=False,
        )
        for prof in out["demographic_profile"]
    ]
    return out


def build_ft_corpus(
    panel: pd.DataFrame, keep_pids: pd.Series, items, fq, language: str
) -> list[dict]:
    hist = panel[panel["syear"].isin(HISTORY_YEARS) & panel["pid"].isin(keep_pids)].copy()
    hist = add_labels(hist)

    examples: list[dict] = []
    for _, row in hist.iterrows():
        profile = render_profile(row)
        for item in items:
            if item.name not in row.index:
                continue
            val = row[item.name]
            if pd.isna(val):
                continue
            if item.value_labels:
                choices = [item.value_labels[k] for k in sorted(item.value_labels)]
                codes = sorted(item.value_labels)
                try:
                    idx = codes.index(int(round(float(val))))
                except (ValueError, TypeError):
                    continue
            else:
                choices = [str(i) for i in range(11)]
                try:
                    idx = int(round(float(val)))
                except (TypeError, ValueError):
                    continue
                if not 0 <= idx <= 10:
                    continue
            examples.append(
                build_ft_example(
                    profile, item.question_de, choices,
                    chr(ord("A") + idx), language=language,
                )
            )
    log.info("  fine-tuning corpus: %d examples from %d person-waves",
             len(examples), len(hist))
    return examples


def apply_design(df: pd.DataFrame, design: str, focal_key: str, seed: int) -> pd.DataFrame:
    if design == "full":
        return df
    if design != "mirror":
        raise typer.BadParameter("design must be 'mirror' or 'full'")
    n = ATP_MIRROR_N[focal_key]
    if len(df) <= n:
        log.warning("Only %d cases available, fewer than the ATP mirror size %d.", len(df), n)
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


@app.command()
def main(
    soep_dir: Path = typer.Option(..., "--soep-dir", help="Directory holding the SOEP files."),
    focal: str = typer.Option("q1", "--focal", help="Focal question key: q1 or q2."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Dataset directory, e.g. datasets/soep_q1."),
    design: str = typer.Option("mirror", "--design", help="'mirror' (ATP sample size) or 'full'."),
    language: str = typer.Option("de", "--language", help="'de' (main) or 'en' (ablation)."),
    history_variant: str = typer.Option(
        "full", "--history-variant",
        help="'full' or 'clean' (drops history items that leak the focal topic).",
    ),
    require_history: bool = typer.Option(True, help="Keep only respondents with >=1 history wave."),
    exclude_special_samples: bool = typer.Option(True, help="Drop non-representative sub-samples."),
    seed: int = typer.Option(42, help="Seed for the mirror subsample."),
    verbose: bool = typer.Option(True),
):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )
    if focal not in FOCAL_QUESTIONS:
        raise typer.BadParameter(f"focal must be one of {list(FOCAL_QUESTIONS)}")
    fq = FOCAL_QUESTIONS[focal]
    items = history_items(history_variant, focal)

    log.info("Focal question %s: %s (%s), wave T = %d", focal, fq.variable, fq.domain, TARGET_YEAR)
    log.info("History waves %s, %d items (variant=%s)", HISTORY_YEARS, len(items), history_variant)

    panel = prepare_panel(soep_dir)

    log.info("Sample construction:")
    target = select_target_wave(panel, fq.variable, exclude_special_samples)
    flow = list(target.attrs.get("flow", []))
    target = attach_history(target, panel, items, require_history)
    flow.append(("history requirement", len(target)))
  
    target["i11102"] = recode.winsorize(target["i11102"])
    target = recode.add_equivalised_income(target)
    target = recode.add_subgroup_vars(target)
    target = add_labels(target)
    target = recode.encode_focal(target, fq)
    target = target[target["correct_answer"].notna()]
    flow.append(("valid answer letter", len(target)))

    target = apply_design(target, design, focal, seed)
    flow.append((f"design={design}", len(target)))

    target = target.reset_index(drop=True)
    target["id"] = np.arange(len(target))
    target = build_prompts(target, fq, language)

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    keep = (
        ["id", "pid", "syear", "question", "choices", "correct_answer", "focal_raw",
         "demographic_profile", "prompt", "history_text", "n_history_waves", "phrf"]
        + [c.name for c in COVARIATES if c.name in target.columns]
        + [v for v in SUBGROUP_VARS if v in target.columns]
    )
    keep = list(dict.fromkeys(c for c in keep if c in target.columns))
    target[keep].to_csv(data_dir / "human_test.csv", index=False)
    log.info("Wrote %s (%d rows)", data_dir / "human_test.csv", len(target))

    target[["id", "pid", "demographic_profile", "history_text"]].to_csv(
        data_dir / "persona_input.csv", index=False
    )

    examples = build_ft_corpus(panel, target["pid"], items, fq, language)
    with (data_dir / "human_train.chat.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    y = recode.phi(target["correct_answer"], fq)
    dist = target["correct_answer"].value_counts().sort_index()
    report = {
        "focal_key": focal,
        "focal_variable": fq.variable,
        "target_year": TARGET_YEAR,
        "history_years": list(HISTORY_YEARS),
        "history_variant": history_variant,
        "language": language,
        "design": design,
        "seed": seed,
        "n_final": int(len(target)),
        "sample_flow": [{"step": s, "n": int(n)} for s, n in flow],
        "answer_distribution": {k: int(v) for k, v in dist.items()},
        "theta_star": float(np.nanmean(y)),
        "sd": float(np.nanstd(y, ddof=1)),
        "skewness": float(pd.Series(y).skew()),
        "n_dropped_by_phi": int(np.isnan(y).sum()),
        "atp_counterpart": fq.atp_counterpart,
        "atp_theta_star": fq.atp_theta_star,
        "n_ft_examples": len(examples),
        "mean_history_waves": float(target["n_history_waves"].mean()),
        "subgroup_sizes": {
            v: {str(k): int(c) for k, c in target[v].value_counts().sort_index().items()}
            for v in SUBGROUP_VARS if v in target.columns
        },
    }
    res_dir = out_dir / "outputs" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("theta* = %.4f  (ATP counterpart: %.2f)", report["theta_star"], fq.atp_theta_star)
    log.info("skewness = %+.3f", report["skewness"])
    log.info("Report: %s", res_dir / "preprocessing_report.json")


if __name__ == "__main__":
    app()
