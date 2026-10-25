import glob, numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree

M_PER_MI = 1609.34
tf = Transformer.from_crs(4326, 32617, always_xy=True)   # lon/lat -> meters (UTM 17N covers your area)

def load_fac(pattern):
    hits = glob.glob(f"**/{pattern}.shp", recursive=True)
    if not hits:
        raise SystemExit(f"No shapefile found for {pattern}")
    g = gpd.read_file(hits[0])
    print(f"{pattern}: {len(g)} features | geometry {g.geom_type.unique().tolist()} | CRS {g.crs}")
    g = g.to_crs(32617)
    g["geometry"] = g.geometry.centroid                   # in case any are polygons
    print("   extent lon/lat:", [round(v, 3) for v in g.to_crs(4326).total_bounds])
    return np.c_[g.geometry.x, g.geometry.y]

fac = {"hospital": load_fac("Hospitals*"), "fire": load_fac("Fire_Stations*"), "ems": load_fac("Ambulance*")}

# Customers: coordinates stay in memory only. No address or road columns are loaded.
c = pd.read_csv("Customer_Clip_attributes.csv", usecols=["circuit_id", "critical_c", "service_po", "service__1"])
c = c.dropna(subset=["service_po", "service__1"])
c = c[(c["service_po"] != 0) & (c["service__1"] != 0)]
c["x"], c["y"] = tf.transform(c["service_po"].values, c["service__1"].values)
print(f"\nCustomers with coordinates: {len(c):,} | lon {c['service_po'].min():.2f} to {c['service_po'].max():.2f}"
      f" | lat {c['service__1'].min():.2f} to {c['service__1'].max():.2f}")

xy = c[["x", "y"]].values
for k, pts in fac.items():
    c[f"{k}_mi"] = cKDTree(pts).query(xy)[0] / M_PER_MI
c["critical"] = c["critical_c"].notna()

LIM = {"hospital": 5, "fire": 3, "ems": 5}               # "near" thresholds in miles (placeholders)
agg = {"cust_with_coords": ("x", "size"), "critical_n": ("critical", "sum")}
for k in fac:
    agg[f"{k}_min_mi"] = (f"{k}_mi", "min")
g = c.groupby("circuit_id").agg(**agg)
for k, lim in LIM.items():
    g[f"{k}_near_share"] = c.assign(n=c[f"{k}_mi"] <= lim).groupby("circuit_id")["n"].mean().round(2)
g["priority_flag"] = (g["critical_n"] > 0) | (g["hospital_min_mi"] <= 1)    # placeholder rule
print("\nDistance to nearest facility across all customers (miles):")
print(c[[f"{k}_mi" for k in fac]].describe(percentiles=[.1, .5, .9]).round(1).to_string())
print("\nCircuits with a priority flag:", int(g["priority_flag"].sum()), "of", len(g))

try:                                                      # attach to the risk table
    g = pd.read_csv("circuit_risk_final.csv", index_col=0).join(g, how="left")
except FileNotFoundError:
    pass
g.to_csv("circuit_priority.csv")                          # aggregates only, safe for the app

# For the API: band for one selected point, with no coordinates returned
trees = {k: cKDTree(p) for k, p in fac.items()}
def priority_bands(lon, lat):
    x, y = tf.transform(lon, lat)
    out = {}
    for k, t in trees.items():
        mi = t.query([x, y])[0] / M_PER_MI
        out[k] = "within 1 mile" if mi <= 1 else "1 to 3 miles" if mi <= 3 else "3 to 5 miles" if mi <= 5 else "more than 5 miles"
    return out