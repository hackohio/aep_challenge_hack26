import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = lambda f, cols=None: pd.read_csv(f, low_memory=False, usecols=cols)
n = lambda s: s.astype(str).str.strip().str.upper()

arcos = R("arcos_prod_damage_09_28_Clip_attributes.csv")
xfmr = R("Transformer_Clip_attributes.csv", ["circuit_id", "aep_circui", "aep_custom", "mounting"])
prim = R("Primary_ExportFeatures_attributes.csv", ["circuit_id", "aep_circui", "mounting", "report_len"])
fuse = R("Fuse_Bank_Clip_attributes.csv", ["circuit_id"])
rec  = R("Recloser_Bank_Clip_attributes.csv", ["circuit_id"])
sw   = R("Switch_Clip_attributes.csv", ["circuit_id"])

# circuit names -> circuit_id
bridge = pd.concat([xfmr[["circuit_id", "aep_circui"]], prim[["circuit_id", "aep_circui"]]]).dropna().drop_duplicates()
bridge["key"] = n(bridge["aep_circui"])
bridge = bridge.drop_duplicates("key")[["key", "circuit_id"]]
arcos["key"] = n(arcos["Circuit"])
arcos = arcos.merge(bridge, on="key", how="left").dropna(subset=["circuit_id"])
arcos["circuit_id"] = arcos["circuit_id"].astype(int)

# storm events (a gap of more than 2 days starts a new event)
arcos["t0"] = pd.to_datetime(arcos["Report_Tim"], errors="coerce")
arcos["t1"] = pd.to_datetime(arcos["Repaired_D"], errors="coerce")
arcos = arcos.dropna(subset=["t0"])
d = arcos["t0"].dt.normalize()
uniq = d.drop_duplicates().sort_values().reset_index(drop=True)
arcos["event_id"] = d.map(dict(zip(uniq, (uniq.diff().dt.days.fillna(99) > 2).cumsum())))
ev = arcos.groupby("event_id").agg(start=("t0", "min"), tickets=("t0", "size"))
ev["severity"] = pd.cut(ev["tickets"], [0, 50, 300, np.inf], labels=["Mild", "Moderate", "Severe"])

# exposure features per circuit
xfmr["oh"] = n(xfmr["mounting"]).eq("OVERHEAD")
prim["oh_len"] = prim["report_len"].where(n(prim["mounting"]).eq("OVERHEAD"), 0)
X = pd.concat([
    xfmr.groupby("circuit_id").agg(xfmr_n=("oh", "size"), cust_n=("aep_custom", "sum"), xfmr_oh_share=("oh", "mean")),
    prim.groupby("circuit_id").agg(prim_len=("report_len", "sum"), prim_oh_len=("oh_len", "sum")),
    fuse.groupby("circuit_id").size().rename("fuse_n"),
    rec.groupby("circuit_id").size().rename("recloser_n"),
    sw.groupby("circuit_id").size().rename("switch_n")], axis=1).fillna(0)
X["oh_line_share"] = (X["prim_oh_len"] / X["prim_len"].replace(0, np.nan)).fillna(0)
for c in ["xfmr_n", "cust_n", "prim_len", "prim_oh_len", "fuse_n", "recloser_n", "switch_n"]:
    X["log_" + c] = np.log1p(X[c])

# circuit x event table, using ALL events; severity is a model input
hit = arcos.groupby(["circuit_id", "event_id"]).size().rename("tix").reset_index()
P = pd.MultiIndex.from_product([X.index, ev.index], names=["circuit_id", "event_id"]).to_frame(index=False)
P = P.merge(hit, how="left", on=["circuit_id", "event_id"])
P["hit"] = P["tix"].notna().astype(int)
P = P.sort_values(["circuit_id", "event_id"]).reset_index(drop=True)
P["prior_hits"] = P.groupby("circuit_id")["hit"].cumsum() - P["hit"]       # earlier storms only
P["hist"] = (P["prior_hits"] + 0.5) / (P.groupby("circuit_id").cumcount() + 2)
P["log_sev"] = np.log(P["event_id"].map(ev["tickets"]))
P = P.merge(X, left_on="circuit_id", right_index=True)

print("Actual share of circuits hit, by storm size:")
print(P.assign(sev=P["event_id"].map(ev["severity"])).groupby("sev", observed=True)["hit"].mean().round(2).to_string())

FULL = ["log_xfmr_n", "log_cust_n", "log_prim_len", "log_prim_oh_len", "log_fuse_n", "log_recloser_n",
        "log_switch_n", "oh_line_share", "xfmr_oh_share"]
SETS = {
    "H history only":              ["hist", "log_sev"],
    "A size only (transformers)":  ["log_xfmr_n", "log_sev"],
    "B size + overhead share":     ["log_prim_len", "log_cust_n", "oh_line_share", "log_sev"],
    "C = B + past-hit history":    ["log_prim_len", "log_cust_n", "oh_line_share", "hist", "log_sev"],
    "D all features + history":    FULL + ["hist", "log_sev"],
}
allc = sorted({c for v in SETS.values() for c in v})
mu, sd = P[allc].mean(), P[allc].std().replace(0, 1)
Z = P.copy()
Z[allc] = (P[allc] - mu) / sd

# leave-one-storm-out backtest on the big storms
print("\nBACKTEST (AUC: 0.5 = coin flip). Trained on all OTHER storms, tested on the one left out.")
for e in ev.index[ev["tickets"] >= 150]:
    for name, cols in SETS.items():
        tr, te = Z[Z["event_id"] != e], Z[Z["event_id"] == e]
        m = LogisticRegression(max_iter=2000, C=0.5).fit(tr[cols], tr["hit"])
        p = m.predict_proba(te[cols])[:, 1]
        print(f"event {e} ({ev.loc[e, 'severity']}) {name:28s} AUC {roc_auc_score(te['hit'], p):.2f} | "
              f"predicted share hit {p.mean():.2f} vs actual {te['hit'].mean():.2f}")

# final model (set C) and a risk table that feeds the severity slider
cols = SETS["C = B + past-hit history"]
m = LogisticRegression(max_iter=2000, C=0.5).fit(Z[cols], Z["hit"])
print("\nSet C coefficients (standardized):\n", pd.Series(m.coef_[0], cols).round(2).sort_values().to_string())

now = X.copy()
now["hist"] = (P.groupby("circuit_id")["hit"].sum().reindex(X.index).fillna(0) + 0.5) / (len(ev) + 2)
reps = ev.groupby("severity", observed=True)["tickets"].median()
print("\nRepresentative tickets per severity:", reps.to_dict())
out = X[["xfmr_n", "cust_n"]].copy()
for sev, t in reps.items():
    F = now.copy()
    F["log_sev"] = np.log(t)
    out["p_" + str(sev).lower()] = m.predict_proba((F[cols] - mu[cols]) / sd[cols])[:, 1].round(3)
out.to_csv("circuit_risk.csv")   # counts and probabilities only, safe for the app
print("\nPredicted share of circuits hit:\n", out.filter(like="p_").describe().round(2).to_string())

# repair timing by storm size, and a restoration curve for the biggest storm
arcos["days"] = (arcos["t1"] - arcos["t0"]).dt.days
arcos["severity"] = arcos["event_id"].map(ev["severity"])
done = arcos[(arcos["Repaired"] == "Yes") & arcos["days"].between(0, 30)]
print("\nRepair days by storm severity:")
print(done.groupby("severity", observed=True)["days"].describe(percentiles=[.5, .75, .9]).round(1).to_string())
top = ev["tickets"].idxmax()
e = arcos[(arcos["event_id"] == top) & (arcos["Repaired"] == "Yes")]
curve = (e["t1"] - e["t0"].min()).dt.days.clip(lower=0).value_counts().sort_index().cumsum() / len(e)
print("\nBiggest storm: share of repaired tickets done by day N after storm start:")
print(curve.round(2).head(12).to_string())