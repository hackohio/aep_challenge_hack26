import pandas as pd, numpy as np

R = lambda f, cols=None: pd.read_csv(f, low_memory=False, usecols=cols)
n = lambda s: s.astype(str).str.strip().str.upper()

arcos = R("arcos_prod_damage_09_28_Clip_attributes.csv")
xfmr = R("Transformer_Clip_attributes.csv", ["location_n", "upstream_l", "aep_circui", "aep_custom"])
fuse = R("Fuse_Bank_Clip_attributes.csv", ["location_n", "upstream_l", "aep_custom"])
rec  = R("Recloser_Bank_Clip_attributes.csv", ["location_n", "upstream_l", "aep_custom"])
sw   = R("Switch_Clip_attributes.csv", ["location_n"])
prim = R("Primary_ExportFeatures_attributes.csv", ["location_n"])
pole = R("Pole_Clip_attributes.csv", ["location_n"])
cust = R("Customer_Clip_attributes.csv", ["aep_xfmr_l"])   # no address or coordinate columns

print("=== A. ARCOS dates ===")
arcos["t0"] = pd.to_datetime(arcos["Report_Tim"], errors="coerce")
arcos["t1"] = pd.to_datetime(arcos["Repaired_D"], errors="coerce")
print("report dates:", arcos["t0"].min().date(), "to", arcos["t0"].max().date())
print("most common repaired dates:\n", arcos["t1"].value_counts().head(5).to_string())
arcos["days"] = (arcos["t1"] - arcos["t0"]).dt.days
done = arcos[arcos["Repaired"] == "Yes"]
print("\nrepair days by work type:")
print(done.groupby("Inform_Wor")["days"].describe()[["count", "min", "25%", "50%", "75%", "max"]].round(1).to_string())
per_day = arcos.groupby("t0").size().sort_values(ascending=False)
print("\ntickets per report date (top 8):\n", per_day.head(8).to_string())
print("report dates with 20+ tickets:", int((per_day >= 20).sum()), "of", len(per_day))

print("\n=== B. ARCOS circuit names not found in Transformer ===")
known = set(n(xfmr["aep_circui"].dropna()))
print(sorted(set(n(arcos["Circuit"].dropna())) - known))

print("\n=== C. What does upstream_l point to? (% found in each table's location_n) ===")
tables = {"Transformer": xfmr, "Fuse": fuse, "Recloser": rec, "Switch": sw, "Primary": prim, "Pole": pole}
locs = {k: set(n(v["location_n"].dropna())) for k, v in tables.items()}
for name, d in [("Transformer", xfmr), ("Fuse", fuse), ("Recloser", rec)]:
    vals = set(n(d["upstream_l"].dropna()))
    print(name, {k: round(100 * len(vals & s) / len(vals), 1) for k, s in locs.items()})

up = pd.concat([xfmr, fuse, rec])[["location_n", "upstream_l"]].dropna()
parent = dict(zip(n(up["location_n"]), n(up["upstream_l"])))
def trace(x):
    seen = set()
    while x in parent and x not in seen:
        seen.add(x)
        x = parent[x]
    return len(seen), x in seen          # (hops to top of chain, ended in a loop?)
res = [trace(x) for x in n(xfmr["location_n"].dropna())]
print("\nhops from transformer to top of chain:\n", pd.Series([r[0] for r in res]).describe().round(1).to_string())
print("chains that loop back on themselves:", sum(r[1] for r in res))

print("\n=== D. Is aep_custom a customer count? ===")
counted = cust.groupby(n(cust["aep_xfmr_l"])).size().rename("counted")
stated = xfmr.assign(k=n(xfmr["location_n"])).drop_duplicates("k").set_index("k")["aep_custom"].rename("stated")
j = pd.concat([counted, stated], axis=1).dropna()
print("transformers compared:", len(j), "| exact match:", round((j["counted"] == j["stated"]).mean() * 100, 1), "%")
print("rank correlation:", round(j.corr(method="spearman").iloc[0, 1], 2))