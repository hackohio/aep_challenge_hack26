import pandas as pd, numpy as np

R = lambda f, cols: pd.read_csv(f, low_memory=False, usecols=cols)
n = lambda s: s.astype(str).str.strip().str.upper()

xf = R("Transformer_Clip_attributes.csv", ["location_n", "upstream_l", "aep_custom", "circuit_id"])
fu = R("Fuse_Bank_Clip_attributes.csv", ["location_n", "upstream_l", "circuit_id"])
rc = R("Recloser_Bank_Clip_attributes.csv", ["location_n", "upstream_l", "circuit_id"])
for d in (xf, fu, rc):
    d["loc"], d["up"] = n(d["location_n"]), n(d["upstream_l"])

# 1) one parent per location (a pole can carry several devices)
links = pd.concat([xf[["loc", "up"]], fu[["loc", "up"]], rc[["loc", "up"]]]).dropna()
links = links[(links["up"] != "NAN") & (links["loc"] != "NAN")]
conflicts = int((links.groupby("loc")["up"].nunique() > 1).sum())
parent = links.drop_duplicates("loc").set_index("loc")["up"].to_dict()
print(f"Locations with a parent: {len(parent):,} | locations with conflicting parents: {conflicts:,}")

# 2) break loops (each device has one parent, so any loop is a closed cycle)
color, flagged, broken = {}, set(), 0
for s in list(parent):
    path, x = [], s
    while x in parent and color.get(x, 0) == 0:
        color[x] = 1
        path.append(x)
        x = parent[x]
    if color.get(x, 0) == 1:                     # the walk closed on itself
        del parent[path[-1]]
        flagged.update(path[path.index(x):])
        broken += 1
    for p in path:
        color[p] = 2
print(f"Loops broken: {broken} | locations inside a loop: {len(flagged)}")

# 3) depth, customers behind each node
nodes = set(parent) | set(parent.values()) | set(xf["loc"]) | set(fu["loc"]) | set(rc["loc"])
depth = {}
def get_depth(x):
    stack = []
    while x not in depth:
        if x not in parent:
            depth[x] = 0
            break
        stack.append(x)
        x = parent[x]
    d = depth[x]
    for y in reversed(stack):
        d += 1
        depth[y] = d
    return depth
for x in nodes:
    get_depth(x)

own = xf.groupby("loc")["aep_custom"].sum().to_dict()
down = {x: own.get(x, 0) for x in nodes}
for x in sorted(nodes, key=lambda k: -depth[k]):  # deepest first, pushing totals up the chain
    if x in parent:
        down[parent[x]] += down[x]

prot = set(fu["loc"].dropna()) | set(rc["loc"].dropna())
def nearest_protector(x):
    x = parent.get(x)                              # start one step above the transformer
    while x is not None:
        if x in prot:
            return x
        x = parent.get(x)
    return None

# 4) per-transformer view: hops to top, customers behind the nearest fuse or recloser
xf["hops"] = xf["loc"].map(depth)
xf["prot"] = xf["loc"].map(nearest_protector)
xf["cust_behind_prot"] = xf["prot"].map(down)
print("\nHops from transformer to the top of its chain:\n", xf["hops"].describe().round(1).to_string())
print("Transformers with no fuse or recloser above them:", round(100 * xf["prot"].isna().mean(), 1), "%")
print("Customers behind the nearest protective device (transformers that have one):")
print(xf["cust_behind_prot"].describe(percentiles=[.5, .9, .99]).round(0).to_string())
print("Biggest 'one device' impact:", int(max(down.values())), "customers")

# 5) topology table for the simulator (backend only; do not expose these keys to customers)
topo = pd.DataFrame({"loc": list(nodes)})
topo["parent"] = topo["loc"].map(parent)
topo["depth"] = topo["loc"].map(depth)
topo["customers_below"] = topo["loc"].map(down)
topo["has_protector"] = topo["loc"].isin(prot)
topo["in_loop"] = topo["loc"].isin(flagged)
topo.to_csv("CSVs/topology.csv", index=False)
print("\nSaved topology.csv:", topo.shape)