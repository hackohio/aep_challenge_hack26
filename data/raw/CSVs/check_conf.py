import pandas as pd, numpy as np
r = pd.read_csv("circuit_risk_final.csv", index_col=0)
x = pd.read_csv("circuit_exposure.csv", index_col=0)
d = r[["xfmr_n", "cust_n", "p_severe_cal", "band_severe", "conf_severe"]].join(x[["prim_len", "oh_line_share"]])
d["line_per_xfmr"] = d["prim_len"] / d["xfmr_n"].replace(0, np.nan)
print("Circuit 13:\n", d.loc[13].round(3).to_string())
print("\nMedian line per transformer across circuits:", round(d["line_per_xfmr"].median(), 2))

d["size_rank"] = d["xfmr_n"].rank(pct=True)
d["risk_rank"] = d["p_severe_cal"].rank(pct=True)
d["gap"] = d["size_rank"] - d["risk_rank"]
sus = d[(d["gap"] > 0.35) & (d["band_severe"] == "Low")]
print(f"\nCircuits where risk ranks far below size, rated Low at Severe: {len(sus)}")
print(sus[["xfmr_n", "prim_len", "line_per_xfmr", "p_severe_cal", "conf_severe"]].round(2).to_string())