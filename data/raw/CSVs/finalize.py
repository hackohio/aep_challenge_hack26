import pandas as pd, numpy as np
from scipy.optimize import brentq

# ---- Part 1: can each storm's repair dates be trusted? ----
a = pd.read_csv("arcos_prod_damage_09_28_Clip_attributes.csv", usecols=["Report_Tim", "Repaired_D"])
a["t0"] = pd.to_datetime(a["Report_Tim"], errors="coerce")
a["t1"] = pd.to_datetime(a["Repaired_D"], errors="coerce")
a = a.dropna(subset=["t0"])
d = a["t0"].dt.normalize()
u = d.drop_duplicates().sort_values().reset_index(drop=True)
a["event_id"] = d.map(dict(zip(u, (u.diff().dt.days.fillna(99) > 2).cumsum())))
a["days"] = (a["t1"] - a["t0"]).dt.days

rows = []
for e, g in a.groupby("event_id"):
    if len(g) < 10:
        continue
    top = g["t1"].dt.normalize().value_counts()
    share = top.iloc[0] / len(g) if len(top) else np.nan
    rows.append({"start": g["t0"].min().date(), "tickets": len(g),
                 "top_repair_date": top.index[0].date() if len(top) else None,
                 "share_on_top_date": round(share, 2),
                 "median_days": g["days"].median(),
                 "verdict": "LIKELY BULK CLOSE" if share >= 0.4 else "plausible"})
print(pd.DataFrame(rows).to_string(index=False))

# ---- Part 2: recalibrate, band, and add confidence ----
risk = pd.read_csv("circuit_risk.csv", index_col=0)
observed = {"mild": 0.03, "moderate": 0.36, "severe": 0.54}   # actual share of circuits hit (from build_v3)
logit = lambda p: np.log(p / (1 - p))
sig = lambda x: 1 / (1 + np.exp(-x))
LOW, HIGH = 0.15, 0.50                                          # band cutoffs: placeholders, adjust after viewing

small = risk["xfmr_n"] < risk["xfmr_n"].quantile(0.2)
nodata = (risk["xfmr_n"] == 0) | (risk["cust_n"] == 0)

for sev, target in observed.items():
    p = risk[f"p_{sev}"].clip(1e-4, 1 - 1e-4)
    shift = brentq(lambda s: sig(logit(p) + s).mean() - target, -10, 10)
    pc = sig(logit(p) + shift)
    risk[f"p_{sev}_cal"] = pc.round(3)
    risk[f"in10_{sev}"] = (pc * 10).round().astype(int)         # "about N in 10 circuits like yours"
    band = pd.cut(pc, [-1, LOW, HIGH, 2], labels=["Low", "Medium", "High"]).astype(object)
    margin = np.minimum((pc - LOW).abs(), (pc - HIGH).abs())
    conf = np.where(small | (margin < 0.05), "Lower", np.where(margin >= 0.10, "Higher", "Medium")).astype(object)
    band[nodata] = "Not rated"
    conf[nodata] = "Insufficient data"
    risk[f"band_{sev}"], risk[f"conf_{sev}"] = band, conf
    print(f"\n{sev.upper()}: shift {shift:+.2f} | mean calibrated {pc.mean():.2f} (target {target})")
    print(pd.crosstab(risk[f"band_{sev}"], risk[f"conf_{sev}"]).to_string())

print("\nCircuits flagged small:", int(small.sum()), "| no data:", int(nodata.sum()), "| total:", len(risk))
risk.to_csv("circuit_risk_final.csv")   # counts, bands, confidence only; safe for the app