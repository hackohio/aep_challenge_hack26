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

# 1) attach circuit_id to ARCOS tickets
bridge = pd.concat([xfmr[["circuit_id", "aep_circui"]], prim[["circuit_id", "aep_circui"]]]).dropna().drop_duplicates()
bridge["key"] = n(bridge["aep_circui"])
bridge = bridge.drop_duplicates("key")[["key", "circuit_id"]]
arcos["key"] = n(arcos["Circuit"])
arcos = arcos.merge(bridge, on="key", how="left")
print("Tickets matched to a circuit:", round(arcos["circuit_id"].notna().mean() * 100, 1), "%")
print("Unmatched circuit names:\n", arcos.loc[arcos["circuit_id"].isna(), "Circuit"].value_counts().to_string())
arcos = arcos.dropna(subset=["circuit_id"]).astype({"circuit_id": int})

# 2) group report dates into storm events (a gap of more than 2 days starts a new event)
arcos["t0"] = pd.to_datetime(arcos["Report_Tim"], errors="coerce")
arcos["t1"] = pd.to_datetime(arcos["Repaired_D"], errors="coerce")
arcos = arcos.dropna(subset=["t0"])
d = arcos["t0"].dt.normalize()
uniq = d.drop_duplicates().sort_values().reset_index(drop=True)
event_of = dict(zip(uniq, (uniq.diff().dt.days.fillna(99) > 2).cumsum()))
arcos["event_id"] = d.map(event_of)

ev = arcos.groupby("event_id").agg(start=("t0", "min"), end=("t0", "max"),
                                   tickets=("t0", "size"), circuits_hit=("circuit_id", "nunique"))
ev["severity"] = pd.cut(ev["tickets"], [0, 50, 300, np.inf], labels=["Mild", "Moderate", "Severe"])
print("\nEVENTS:\n", ev.sort_values("tickets", ascending=False).head(12).to_string())

# 3) repair days by work type (day resolution only, clipped at 30 days)
arcos["days"] = (arcos["t1"] - arcos["t0"]).dt.days
done = arcos[arcos["Repaired"] == "Yes"]
print("\nShare of repaired tickets taking >30 days:", round((done["days"] > 30).mean() * 100, 1), "%")
print(done[done["days"].between(0, 30)].groupby("Inform_Wor")["days"]
      .describe(percentiles=[.25, .5, .75, .9]).round(1).to_string())

# 4) exposure features per circuit
xfmr["oh"] = n(xfmr["mounting"]).eq("OVERHEAD")
prim["oh"] = n(prim["mounting"]).eq("OVERHEAD")
prim["oh_len"] = prim["report_len"].where(prim["oh"], 0)
X = pd.concat([
    xfmr.groupby("circuit_id").agg(xfmr_n=("oh", "size"), cust_n=("aep_custom", "sum"), xfmr_oh_share=("oh", "mean")),
    prim.groupby("circuit_id").agg(prim_len=("report_len", "sum"), prim_oh_len=("oh_len", "sum")),
    fuse.groupby("circuit_id").size().rename("fuse_n"),
    rec.groupby("circuit_id").size().rename("recloser_n"),
    sw.groupby("circuit_id").size().rename("switch_n"),
], axis=1).fillna(0)
X["oh_line_share"] = (X["prim_oh_len"] / X["prim_len"].replace(0, np.nan)).fillna(0)

# 5) circuit x event table: was the circuit hit?
big = ev.index[ev["tickets"] >= 50]
hit = arcos.groupby(["circuit_id", "event_id"]).size().rename("tickets").reset_index()
panel = pd.MultiIndex.from_product([X.index, big], names=["circuit_id", "event_id"]).to_frame(index=False)
panel = panel.merge(hit, how="left", on=["circuit_id", "event_id"])
panel["hit"] = (panel["tickets"].fillna(0) > 0).astype(int)
panel = panel.merge(X, left_on="circuit_id", right_index=True)
print("\nShare of circuits hit, by event:\n", panel.groupby("event_id")["hit"].mean().round(2).to_string())

# 6) leave-one-event-out backtest vs. a "bigger circuit = more damage" baseline
size_cols = ["xfmr_n", "cust_n", "prim_len", "prim_oh_len", "fuse_n", "recloser_n", "switch_n"]
cols = size_cols + ["oh_line_share", "xfmr_oh_share"]
Z = panel.copy()
Z[size_cols] = np.log1p(Z[size_cols])
Z[cols] = (Z[cols] - Z[cols].mean()) / Z[cols].std().replace(0, 1)
print("\nBACKTEST (AUC: 0.5 = coin flip, 1.0 = perfect)")
for e in big:
    tr, te = Z[Z["event_id"] != e], Z[Z["event_id"] == e]
    if te["hit"].nunique() < 2 or tr["hit"].nunique() < 2:
        print(f"event {e}: skipped (all or none hit)"); continue
    m = LogisticRegression(max_iter=1000, C=0.5).fit(tr[cols], tr["hit"])
    print(f"event {e}: model AUC {roc_auc_score(te['hit'], m.predict_proba(te[cols])[:, 1]):.2f} | "
          f"baseline (transformer count) AUC {roc_auc_score(te['hit'], te['xfmr_n']):.2f}")

m = LogisticRegression(max_iter=1000, C=0.5).fit(Z[cols], Z["hit"])
print("\nFactors pushing risk up (+) or down (-):\n", pd.Series(m.coef_[0], cols).sort_values().round(2).to_string())

X.to_csv("circuit_exposure.csv"); ev.to_csv("events.csv")   # counts only, safe for the app