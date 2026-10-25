import pandas as pd, numpy as np
from collections import Counter

# ---- parse matched_upstream.txt ----
rows = []
with open("matched_upstream.txt") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        prem, xf, devs, brk = parts[0].strip().zfill(9), None, [], None
        for seg in parts[1:]:
            p = seg.split("*")
            if len(p) != 4:
                continue
            t, _, kind, key = p
            if t == "eo_power_xfrmr_inst": xf = key.strip().upper()
            elif t == "eo_isolating_eqpt_inst": devs.append((key.strip().upper(), kind))
            elif t == "eo_circuit": brk = key.strip().upper()
        rows.append((prem, xf, devs, brk))
ch = pd.DataFrame(rows, columns=["prem", "xf", "devs", "brk"])
ch["n_dev"] = ch["devs"].str.len()
print(f"Premises in file: {len(ch):,} | with transformer: {ch['xf'].notna().mean():.1%} | with breaker: {ch['brk'].notna().mean():.1%}")
print("Protective devices per premise:\n", ch["n_dev"].describe().round(1).to_string())
print("Device kinds:", Counter(k for d in ch["devs"] for _, k in d).most_common())

# ---- match to Customer (no address columns loaded) ----
cu = pd.read_csv("/home/avopti/Documents/Codes/Data Analysis_AEP/data/raw/CSVs/Customer_Clip_attributes.csv", usecols=["premise_nu", "circuit_id", "aep_xfmr_l"])
cu["prem"] = cu["premise_nu"].astype(str).str.zfill(9)
cu["xk"] = cu["aep_xfmr_l"].astype(str).str.strip().str.upper()
m = ch.merge(cu[["prem", "circuit_id", "xk"]], on="prem", how="left")
print(f"\nFile premises found in Customer: {m['circuit_id'].notna().mean():.1%} | Customer premises covered by file: {cu['prem'].isin(ch['prem']).mean():.1%}")
print(f"Transformer key agrees with Customer.aep_xfmr_l: {(m['xf'] == m['xk'])[m['circuit_id'].notna()].mean():.1%}")

# ---- breaker code -> circuit_id ----
g = m.dropna(subset=["circuit_id"]).groupby("brk")["circuit_id"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
print(f"Breaker codes: {len(g)} | share of codes mapping to one circuit_id (>=95% of premises): {(g >= .95).mean():.1%}")

# ---- exact customers behind each device ----
cnt = Counter(k for d in ch["devs"] for k, _ in d)
first = ch["devs"].map(lambda d: cnt[d[0][0]] if d else np.nan)
print("\nPremises behind the FIRST device in each chain:\n", first.describe(percentiles=[.5, .9, .99]).round(0).to_string())
mono = ch["devs"].map(lambda d: all(cnt[a[0]] <= cnt[b[0]] for a, b in zip(d, d[1:])) if len(d) > 1 else None).dropna()
print(f"Chains where counts never decrease going up (confirms 'nearest first'): {mono.mean():.1%}")

# ---- compare to the graph from build_graph.py ----
tp = pd.read_csv("/home/avopti/Documents/Codes/Data Analysis_AEP/data/raw/CSVs/topology.csv").set_index("loc")["customers_below"]
cmp = pd.DataFrame({"file": pd.Series(cnt), "graph": tp}).dropna()
print(f"\nDevices in both: {len(cmp):,} | rank correlation file vs graph: {cmp.corr(method='spearman').iloc[0, 1]:.2f}"
      f" | median ratio graph/file: {(cmp['graph'] / cmp['file']).median():.2f}")
print("Top 5 graph nodes by customers_below (looking for the 107k node):")
print(pd.read_csv("/home/avopti/Documents/Codes/Data Analysis_AEP/data/raw/CSVs/topology.csv").nlargest(5, "customers_below")[["depth", "customers_below", "has_protector", "in_loop"]].to_string(index=False))

# ---- transformer-level table for the app (no premise numbers) ----
x = ch.dropna(subset=["xf"]).drop_duplicates("xf").copy()
x["first_dev_kind"] = x["devs"].map(lambda d: d[0][1] if d else None)
x["cust_behind_first_dev"] = x["devs"].map(lambda d: cnt[d[0][0]] if d else np.nan)
x["cust_behind_top_dev"] = x["devs"].map(lambda d: max((cnt[k] for k, _ in d), default=np.nan))
x["cust_on_xfmr"] = x["xf"].map(ch.groupby("xf").size())
cir = m.dropna(subset=["circuit_id"]).groupby("xf")["circuit_id"].first()
x["circuit_id"] = x["xf"].map(cir)
x["n_dev_above"] = x["n_dev"]
x[["xf", "circuit_id", "brk", "n_dev_above", "first_dev_kind", "cust_on_xfmr",
   "cust_behind_first_dev", "cust_behind_top_dev"]].to_csv("CSVs/xfmr_chain.csv", index=False)
print("\nSaved xfmr_chain.csv:", len(x), "transformers")