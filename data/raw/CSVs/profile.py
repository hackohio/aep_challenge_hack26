import pandas as pd, glob, itertools

files = {}
for f in sorted(glob.glob("*.csv")):
    name = f.replace("_attributes.csv", "")
    files[name] = pd.read_csv(f, low_memory=False)

for name, df in files.items():
    print(f"\n=== {name}: {len(df):,} rows x {df.shape[1]} cols")
    for c in df.columns:
        nulls = df[c].isna().mean() * 100
        print(f"  {c:35s} {str(df[c].dtype):10s} nulls {nulls:5.1f}%  unique {df[c].nunique():,}")

print("\n\n=== Shared column names across files (possible join keys) ===")
cols = {n: set(c.lower() for c in df.columns) for n, df in files.items()}
for a, b in itertools.combinations(cols, 2):
    shared = cols[a] & cols[b]
    if shared:
        print(f"{a} <-> {b}: {sorted(shared)}")