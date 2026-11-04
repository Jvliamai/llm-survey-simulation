"""
Extension C figure
(a) bias, lowest value marked by a star, (b) ESS gain at the same number of correction labels.
Lines = mean over the 100 draws of H, bands = mean +- 1 SD.

    python experiments/extension_c_figure.py --focal q1
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BLUE_BAND, BLUE_LINE = "#A6C8E0", "#4F86B5"
ORANGE_BAND, ORANGE_LINE = "#FDB462", "#E08A1E"
GREY = "#6B6B66"
STYLE = {
    "ft": dict(color=BLUE_LINE, band=BLUE_BAND, marker="o", label="Domain-FT"),
    "icl": dict(color=ORANGE_LINE, band=ORANGE_BAND, marker="s", label="ICL"),
}
SHIFT = {"ft": -0.8, "icl": 0.8}


def summarise(tr):
    tr = tr.assign(abs_err=tr["err"].abs())
    g = tr.groupby(["method", "alpha"])
    return g.agg(bias=("abs_err", "mean"), bias_sd=("abs_err", "std"),
                 ess_n=("ess_same_n", "mean"), ess_n_sd=("ess_same_n", "std"),
                 ess_b=("ess_budget", "mean"), ess_b_sd=("ess_budget", "std")).reset_index()


def figure(s, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    panels = [("bias", "bias_sd", "Bias (%)", "(a)"),
              ("ess_n", "ess_n_sd", r"ESS gain, same $n$ (%)", "(b)")]
    refs = s[s["method"].isin(["zeroshot", "human"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for ax, (col, sd, ylab, tag) in zip(axes, panels):
        for m, st in STYLE.items():
            d = s[s["method"] == m].sort_values("alpha")
            x = d["alpha"] * 100 + SHIFT[m]
            ax.fill_between(x, d[col] - d[sd], d[col] + d[sd],
                            color=st["band"], alpha=0.35, lw=0)
            ax.plot(x, d[col], color=st["color"], marker=st["marker"],
                    markersize=3, linewidth=1.2, label=st["label"])
            if col == "bias":
                i = d[col].idxmin()
                ax.plot(d.loc[i, "alpha"] * 100 + SHIFT[m], d.loc[i, col],
                        marker="*", color="black", markersize=8, zorder=5,
                        linestyle="none")
        # Reference lines: unadapted model (grey, solid) and human only (black, dotted).
        # The dotted line is drawn on top so that both remain visible where they coincide.
        for ref, colour, ls, lw, z, lab in (
                ("zeroshot", GREY, "-", 1.6, 2, "No adaptation"),
                ("human", "black", ":", 1.3, 3, "Human only")):
            r = refs[refs["method"] == ref]
            if len(r):
                ax.axhline(r[col].iloc[0], color=colour, ls=ls, lw=lw,
                           zorder=z, label=lab)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 85)
        ax.set_xticks([5, 20, 40, 60, 80])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="#E6E6E6", linestyle="-", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(tag, loc="left", fontweight="bold")
    fig.supxlabel(r"Share of budget used for adaptation, $\alpha$ (%)", fontsize=9, y=0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    fig.savefig(out_dir / "extension_c_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "extension_c_figure.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="q1")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir
               else ROOT / "outputs" / "tables" / f"extc_{args.focal}")
    path = out_dir / "extension_c_trials.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- rerun evaluate first.")
    s = summarise(pd.read_csv(path))
    figure(s, out_dir)
    print(s.round(3).to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()