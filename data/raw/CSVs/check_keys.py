import pandas as pd

R = lambda f, **k: pd.read_csv(f, low_memory=False, **k)
arcos = R("arcos_prod_damage_09_28_Clip_attributes.csv")
xfmr  = R("Transformer_Clip_attributes.csv")
fuse  = R("Fuse_Bank_Clip_attributes.csv")
rec   = R("Recloser_Bank_Clip_attributes.csv")
sw    = R("Switch_Clip_attributes.csv")
prim  = R("Primary_ExportFeatures_attributes.csv")
sec   = R("Secondary_ExportFeatures_attributes.csv")
pole  = R("Pole_Clip_attributes.csv", usecols=["id", "location_n", "material", "height", "class", "date_insta"])
cust  = R("Customer_Clip_attributes.csv", usecols=["circuit_id", "aep_xfmr_l", "aep_locati", "kind_of_se", "service_po", "service__1"])

norm = lambda s: s.dropna().astype(str).str.strip().str.upper()

def overlap(label, a, b):
    a, b = set(norm(a)), set(norm(b))
    print(f"{label:55s} {100*len(a & b)/max(len(a),1):5.1f}% of {len(a):,} found")

print("--- KEY OVERLAPS ---")
overlap("ARCOS.Circuit in Transformer.aep_circui", arcos["Circuit"], xfmr["aep_circui"])
overlap("ARCOS.Substation in Transformer.aep_statio", arcos["Substation"], xfmr["aep_statio"])
overlap("Customer.aep_xfmr_l in Transformer.location_n", cust["aep_xfmr_l"], xfmr["location_n"])
for n, d in [("Pole", pole), ("Secondary", sec)]:
    overlap(f"Customer.aep_locati in {n}.location_n", cust["aep_locati"], d["location_n"])
all_loc = pd.concat([xfmr["location_n"], fuse["location_n"], rec["location_n"],
                     sw["location_n"], prim["location_n"], pole["location_n"]])
for n, d in [("Transformer", xfmr), ("Fuse", fuse), ("Recloser", rec)]:
    overlap(f"{n}.upstream_l in any device location_n", d["upstream_l"], all_loc)
for n, d in [("Pole", pole), ("Transformer", xfmr), ("Primary", prim)]:
    ids = set(d["id"].astype(str))
    print(f"ARCOS.NEAR_FID found in {n}.id: {100*arcos['NEAR_FID'].astype(str).isin(ids).mean():.1f}%")

print("\n--- CATEGORY VALUES ---")
for n, d, cols in [("ARCOS", arcos, ["Asset_Type", "Inform_Wor", "Access_Iss", "Repaired", "Special_Eq", "Media"]),
                   ("Primary", prim, ["mounting", "usage"]),
                   ("Transformer", xfmr, ["device_typ", "mounting"]),
                   ("Fuse", fuse, ["normal_sta"]), ("Switch", sw, ["normal_sta"]),
                   ("Pole", pole, ["material"]), ("Customer", cust, ["kind_of_se"])]:
    for c in cols:
        print(f"{n}.{c}:", d[c].value_counts(dropna=False).head(10).to_dict())

print("\n--- SAMPLES ---")
print(arcos[["Report_Tim", "Repaired_D"]].head(3).to_string())
print(arcos[["NEAR_DIST"]].describe().T.to_string())
print(cust[["service_po", "service__1"]].describe().T[["min", "max"]].to_string())