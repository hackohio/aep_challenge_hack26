import json, random, heapq
import numpy as np

D = json.load(open("chains.json"))
CHAINS, CZ = D["chains"], D["circuit_zones"]

# PLACEHOLDERS: replace with STORMS (person-hours) and indirect-labor values
LABOR = {"wire": (2, 4, 10), "pole": (6, 10, 18), "equipment": (3, 5, 9), "switch": (2, 3, 6)}  # (min, likely, max)
MIX = {"wire": .52, "switch": .215, "pole": .214, "equipment": .049}   # ARCOS work-type shares
TRAVEL, NONPROD, CREW_SIZE = 0.25, 0.20, 2
ASSESS = (2, 4, 10)                  # hours before repairs start (min, likely, max)
ACCESS_RATE, ACCESS_MULT = 0.32, 1.5 # 32% of ARCOS tickets had access issues; 1.5x is a guess
SEV = {"mild": 0.8, "moderate": 1.0, "severe": 1.3}   # scales assessment delay and travel; a guess

def pick_type(dtype):
    return random.choices(list(MIX), list(MIX.values()))[0] if dtype == "mix" else dtype

def hours(dt, sev, access):
    lo, mode, hi = LABOR[dt]
    h = random.triangular(lo, hi, mode) / CREW_SIZE
    h *= (1 + TRAVEL * SEV[sev]) * (1 + NONPROD)
    if access == "on" or (access == "base" and random.random() < ACCESS_RATE):
        h *= ACCESS_MULT
    return h

def one_run(c, dtype, count, crews, sev, access):
    zones = c["zones"]
    on_path = {z[0] for z in zones}
    forced = random.choice(zones)                       # damage on your path (the condition)
    pool = [z for z in CZ[str(c["circuit"])] if z[0] != forced[0]]
    hit = [forced] + random.sample(pool, min(count - 1, len(pool)))
    jobs = sorted(hit, key=lambda z: -z[2])             # most customers first, so upstream goes first
    free = [random.triangular(ASSESS[0], ASSESS[2], ASSESS[1]) * SEV[sev]] * crews
    heapq.heapify(free)
    end = 0.0
    for z in jobs:
        fin = heapq.heappop(free) + hours(pick_type(dtype), sev, access)
        heapq.heappush(free, fin)
        if z[0] in on_path:                             # you are back when every damaged zone on your path is fixed
            end = max(end, fin)
    return end

def simulate(xf, dtype="mix", count=3, crews=2, sev="moderate", access="base", runs=500):
    c = CHAINS.get(xf)
    if c is None:
        return {"fallback": "No network data for this location, so no estimate is shown."}
    t = np.array([one_run(c, dtype, count, crews, sev, access) for _ in range(runs)])
    lo, med, hi = np.percentile(t, [10, 50, 90])
    return {"range_hours": [round(lo), round(hi)], "typical_hours": round(med), "source": "model_estimate"}

def drivers(xf, **kw):                                  # "what moves this estimate"
    base = simulate(xf, **kw)["typical_hours"]
    tests = {"two more crews": dict(crews=kw.get("crews", 2) + 2),
             "no access problems": dict(access="off"),
             "half the damage": dict(count=max(1, kw.get("count", 3) // 2))}
    return {k: round(simulate(xf, **{**kw, **v})["typical_hours"] - base) for k, v in tests.items()}

if __name__ == "__main__":
    xf = "37800182B47663"
    for count, crews in [(1, 1), (10, 1), (10, 5), (30, 5), (30, 15)]:
        print(f"{count} damages, {crews} crews:", simulate(xf, count=count, crews=crews, sev="severe")["range_hours"])
    print("Change in typical hours:", drivers(xf, count=10, crews=2, sev="moderate"))