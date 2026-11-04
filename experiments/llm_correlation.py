"""Direct correlation of LLM predictions with human answers, and covariate eta^2 on Q2."""
import pandas as pd, numpy as np

MAPS = {"soep_q1": dict(zip("ABCDE", [4,3,2,1,0])),
        "soep_q2": dict(zip("ABCDEFGHIJK", range(11)))}
METHODS = {"silicon_sampling": "Demo-only", "concept_guided": "Persona-guided", "ft": "Domain-FT"}
MODELS = ["llama", "ministral", "qwen"]
COVS = ["age","sex","pgcasmin","pgbilzeit","pgemplst","pgstib_group","pgisco08_major",
        "pgnace_section","pglabgro","i11102","pgfamstd","pgpartz","d11106","d11107",
        "pli0096_h","hgowner","bula_h","sampreg","migback","pgnation","plh0258_h",
        "pli0098_h","ple0008","plh0011_h","plh0012_h"]
NUMERIC = ["age","pgbilzeit","pglabgro","i11102","d11106","d11107"]

rows, store = [], {}
for q, m in MAPS.items():
    for model in MODELS:
        for f, name in METHODS.items():
            df = pd.read_csv(f"datasets/{q}/outputs/simulated_data/{model}/{f}.csv")
            y = df["correct_answer"].map(m)
            p = df["answer"].map(m)
            ok = y.notna() & p.notna()
            corr = np.corrcoef(y[ok], p[ok])[0, 1] if p[ok].std() > 0 else np.nan
            rows.append({"question": q, "model": model, "method": name,
                         "valid_share": ok.sum() / y.notna().sum(), "corr": corr,
                         "pred_sd": p[ok].std(), "y_sd": y[ok].std(),
                         "n_categories": p[ok].nunique(),
                         "top_share": p[ok].value_counts(normalize=True).iloc[0]})
            store[(q, model, name)] = (df, y, p)

out = pd.DataFrame(rows)
out.to_csv("outputs/tables/discussion/llm_correlation.csv", index=False)
print("=== Pro Modell und Methode ===")
print(out.round(3).to_string(index=False))
print("\n=== Mittel pro Frage und Methode ===")
print(out.groupby(["question", "method"])[["corr", "valid_share", "pred_sd", "top_share"]].mean().round(3))

def eta2(g, v):
    ok = v.notna() & g.notna()
    grp = g[ok].astype(str); vv = v[ok].astype(float)
    if vv.var() == 0:
        return np.nan
    means = vv.groupby(grp).transform("mean")
    return 1 - ((vv - means) ** 2).sum() / ((vv - vv.mean()) ** 2).sum()

q, m = "soep_q2", MAPS["soep_q2"]
base = pd.read_csv(f"datasets/{q}/data/human_test.csv")
y = base["correct_answer"].map(m)
erows = []
for c in COVS:
    g = base[c]
    if c in NUMERIC:
        g = pd.qcut(g.astype(float), 5, duplicates="drop")
    row = {"covariate": c, "eta2_true": eta2(g, y)}
    for name in METHODS.values():
        vals = [eta2(g, store[(q, model, name)][2]) for model in MODELS]
        row[f"eta2_{name}"] = np.nanmean(vals)
    erows.append(row)
eta = pd.DataFrame(erows).sort_values("eta2_true", ascending=False)
eta.to_csv("outputs/tables/discussion/covariate_eta_q2.csv", index=False)
print("\n=== SOEP Q2: erklaerte Varianz je Kovariate (eta^2), echte Antworten vs. LLM ===")
print(eta.round(3).head(10).to_string(index=False))
