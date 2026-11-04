"""Predictability of the focal questions from the 25 covariates (ridge, 5-fold CV)."""
import pandas as pd, numpy as np

COVS = ["age","sex","pgcasmin","pgbilzeit","pgemplst","pgstib_group","pgisco08_major",
        "pgnace_section","pglabgro","i11102","pgfamstd","pgpartz","d11106","d11107",
        "pli0096_h","hgowner","bula_h","sampreg","migback","pgnation","plh0258_h",
        "pli0098_h","ple0008","plh0011_h","plh0012_h"]
NUMERIC = ["age","pgbilzeit","pglabgro","i11102","d11106","d11107"]
MAPS = {"soep_q1": dict(zip("ABCDE", [4,3,2,1,0])),
        "soep_q2": dict(zip("ABCDEFGHIJK", range(11)))}

def design(df):
    parts = []
    for c in COVS:
        if c in NUMERIC:
            x = df[c].astype(float)
            parts.append(x.fillna(x.median()).rename(c))
            parts.append(x.isna().astype(float).rename(c + "_na"))
        else:
            parts.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True).astype(float))
    X = pd.concat(parts, axis=1)
    return X.loc[:, X.std() > 0].values

def ridge_pred(Xtr, ytr, Xte, a):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    beta = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return ym + B @ beta

rows = []
for q, m in MAPS.items():
    df = pd.read_csv(f"datasets/{q}/data/human_test.csv")
    df["y"] = df["correct_answer"].map(m)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    X, y = design(df), df["y"].values.astype(float)
    n = len(y); idx = np.random.default_rng(0).permutation(n)
    best = None
    for a in np.logspace(0, 4, 17):
        pred = np.empty(n)
        for k in range(5):
            te = idx[k::5]; tr = np.setdiff1d(idx, te)
            pred[te] = ridge_pred(X[tr], y[tr], X[te], a)
        r2 = 1 - np.mean((y - pred)**2) / np.var(y)
        rho = np.corrcoef(y, pred)[0, 1]
        if best is None or r2 > best[1]:
            best = (a, r2, rho)
    rows.append({"question": q, "n": n, "alpha": best[0], "r2_cv": best[1], "corr_cv": best[2]})

out = pd.DataFrame(rows)
out.to_csv("outputs/tables/discussion/predictability.csv", index=False)
print(out.round(3).to_string(index=False))
