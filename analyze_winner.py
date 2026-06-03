"""1着馬を予測でどう捉えていたか集計"""
import sys, json
sys.path.insert(0, ".")
import pandas as pd
from src.features import build_feature_matrix
from src.model import heuristic_score

classic = json.load(open("data/classic_results.json", encoding="utf-8"))

print("{:>6} {:<6} {:>15} {:>10}".format("birth", "race", "winner_pred_rank", "winner_id"))
print("-" * 50)
agg = {"derby": {"top1":0, "top3":0, "top5":0, "top10":0, "top30":0, "n":0, "ranks":[]},
       "oaks":  {"top1":0, "top3":0, "top5":0, "top10":0, "top30":0, "n":0, "ranks":[]}}
for birth in range(2015, 2023):
    for rt in ["derby", "oaks"]:
        top5 = classic.get(rt, {}).get(str(birth), [])
        if not top5: continue
        winner_id = top5[0]
        horses = pd.read_csv(f"data/horses_{birth}.csv")
        sex = "牡" if rt=="derby" else "牝"
        feats = build_feature_matrix(horses, birth_year=birth, sex_filter=sex)
        feats["score"] = heuristic_score(feats, race_type=rt)
        feats = feats.sort_values("score", ascending=False).reset_index(drop=True)
        feats["rank"] = feats.index + 1
        row = feats[feats["horse_id"].astype(str) == winner_id]
        rank = int(row.iloc[0]["rank"]) if not row.empty else None
        print("{:>6} {:<6} {:>15} {:>10}".format(birth, rt, rank if rank else "-", winner_id))
        a = agg[rt]
        a["n"] += 1
        if rank:
            a["ranks"].append(rank)
            if rank <= 1: a["top1"] += 1
            if rank <= 3: a["top3"] += 1
            if rank <= 5: a["top5"] += 1
            if rank <= 10: a["top10"] += 1
            if rank <= 30: a["top30"] += 1

print("\n=== 1着馬を予測のTOP N に捉えた年 (8年中) ===")
print("{:<6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10}".format("race","TOP1","TOP3","TOP5","TOP10","TOP30","平均順位"))
for rt in ["derby","oaks"]:
    a = agg[rt]
    avg = sum(a["ranks"])/len(a["ranks"]) if a["ranks"] else 0
    print("{:<6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10.0f}".format(rt, a["top1"], a["top3"], a["top5"], a["top10"], a["top30"], avg))
