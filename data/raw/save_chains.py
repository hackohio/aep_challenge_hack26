import json, pandas as pd
from collections import Counter, defaultdict

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
            key = key.strip().upper()
            if t == "eo_power_xfrmr_inst": xf = key
            elif t == "eo_isolating_eqpt_inst": devs.append((key, kind))
            elif t == "eo_circuit": brk = key
        rows.append((prem, xf, devs, brk))
ch = pd.DataFrame(rows, columns=["prem", "xf", "devs", "brk"])

cu = pd.read_csv("/home/avopti/Documents/Codes/Data Analysis_AEP/data/raw/CSVs/Customer_Clip_attributes.csv", usecols=["premise_nu", "circuit_id", "aep_xfmr_l"])
cu["prem"] = cu["premise_nu"].astype(str).str.zfill(9)
cu["xk"] = cu["aep_xfmr_l"].astype(str).str.strip().str.upper()
ch = ch.merge(cu[["prem", "circuit_id", "xk"]], on="prem", how="left").dropna(subset=["xf", "circuit_id"])
ch["circuit_id"] = ch["circuit_id"].astype(int)
ch["same_xf"] = ch["xf"] == ch["xk"]

dev_cnt = Counter(k for d in ch["devs"] for k, _ in d)      # premises behind each device
xf_cnt = ch.groupby("xf").size().to_dict()
brk_cnt = ch.groupby("brk").size().to_dict()
agree_by_xf = ch.groupby("xf")["same_xf"].mean().to_dict()

chains, czones = {}, defaultdict(dict)
for r in ch.drop_duplicates("xf").itertuples():
    zones = [[r.xf, "Transformer", xf_cnt[r.xf]]]
    zones += [[k, kind, dev_cnt[k]] for k, kind in r.devs]
    zones.append([r.brk, "Circuit Breaker", brk_cnt[r.brk]])
    chains[r.xf] = {"circuit": r.circuit_id, "agree": round(agree_by_xf[r.xf], 2), "zones": zones}
    for z in zones:
        czones[r.circuit_id][z[0]] = z

json.dump({"chains": chains, "circuit_zones": {str(c): list(v.values()) for c, v in czones.items()}},
          open("chains.json", "w"))
print(f"Saved {len(chains):,} transformers across {len(czones)} circuits | "
      f"{100 * sum(c['agree'] == 1 for c in chains.values()) / len(chains):.0f}% fully agree with Customer")

ok = sorted([(x, c) for x, c in chains.items() if c["agree"] == 1 and len(c["zones"]) >= 4],
            key=lambda t: t[1]["zones"][1][2])
for q in (0.25, 0.5, 0.75, 0.9):
    x, c = ok[int(q * len(ok))]
    print(f"{int(q*100)}th pct demo pick: {x} | circuit {c['circuit']} | "
          f"customers behind first device {c['zones'][1][2]} | zones {len(c['zones'])}")