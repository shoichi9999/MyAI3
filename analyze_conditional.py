"""TOP5予測のうち実際に出走した馬の的中率（条件付き的中率）"""
import sys, json
sys.path.insert(0, ".")
import pandas as pd
from src.features import build_feature_matrix
from src.model import heuristic_score

starters = json.load(open("data/race_starters_cache.json", encoding="utf-8"))
classic = json.load(open("data/classic_results.json", encoding="utf-8"))

DERBY_RID = {2018: "201805021210"}
hdr = ("birth", "race", "pred5_starters", "pred5_top5_hits", "rate%")
print("{:>6} {:<6} {:>14} {:>15} {:>7}".format(*hdr))
print("-" * 60)

sums = {"derby": [0, 0], "oaks": [0, 0]}
for birth in range(2015, 2023):
    ry = birth + 3
    for rt in ["derby", "oaks"]:
        rid = DERBY_RID.get(ry, f"{ry}05021211") if rt == "derby" else f"{ry}05021011"
        ss = set(starters.get(rid, []))
        top5_actual = set(classic.get(rt, {}).get(str(birth), []))
        horses = pd.read_csv(f"data/horses_{birth}.csv")
        sex = "牡" if rt == "derby" else "牝"
        feats = build_feature_matrix(horses, birth_year=birth, sex_filter=sex)
        feats["score"] = heuristic_score(feats, race_type=rt)
        pred5_ids = set(feats.sort_values("score", ascending=False).head(5)["horse_id"].astype(str))
        n_starts = len(pred5_ids & ss)
        n_hits = len(pred5_ids & top5_actual)
        rate = (n_hits / n_starts * 100) if n_starts > 0 else 0.0
        sums[rt][0] += n_starts
        sums[rt][1] += n_hits
        print("{:>6} {:<6} {:>14} {:>15} {:>6.1f}%".format(birth, rt, n_starts, n_hits, rate))

print("-" * 60)
for rt in ["derby", "oaks"]:
    s, h = sums[rt]
    r = (h / s * 100) if s else 0
    print("{:>6} {:<6} {:>14} {:>15} {:>6.1f}%".format("TOTAL", rt, s, h, r))
