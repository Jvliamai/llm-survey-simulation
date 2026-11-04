"""Correlation of allocation predictions with the human answers, per method and alpha."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soepsim.preprocessing.recode import phi  # noqa: E402
from soepsim.preprocessing.variables import FOCAL_QUESTIONS  # noqa: E402

fq = FOCAL_QUESTIONS["q1"]
rows = []
for wdir in sorted((ROOT / "datasets" / "soep_q1_full" / "extc").glob("seed*")):
    frame = pd.read_csv(wdir / "data" / "frame.csv")
    y = phi(frame["correct_answer"], fq)
    for path in sorted((wdir / "predictions").glob("*/*.csv")):
        method, t = path.stem.rsplit("_", 1)
        pred = frame[["id"]].merge(pd.read_csv(path)[["id", "answer"]], on="id", how="left")
        yhat = phi(pred["answer"], fq)
        ok = ~(np.isnan(y) | np.isnan(yhat))
        corr = np.corrcoef(y[ok], yhat[ok])[0, 1] if np.std(yhat[ok]) > 0 else np.nan
        rows.append({"seed": wdir.name, "model": path.parent.name, "method": method,
                     "alpha": int(t[1:]), "n": int(ok.sum()), "corr": corr})

out = pd.DataFrame(rows).sort_values(["method", "alpha"])
dest = ROOT / "outputs" / "tables" / "rq3" / "extension_c_correlation.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False)
print(out.round(3).to_string(index=False))
print(f"\nmax corr: {out['corr'].max():.3f}   written to {dest}")
