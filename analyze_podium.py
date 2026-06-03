"""予測TOP5に「1着」「1〜3着」「1〜5着」の馬が含まれた年数を集計"""
import sys, json
sys.path.insert(0, ".")
import pandas as pd
from src.features import build_feature_matrix
from src.model import heuristic_score

classic = json.load(open("data/classic_results.json", encoding="utf-8"))

agg = {"derby":{"win":0,"podium":0,"top5":0,"n":0,"rows":[]},
       "oaks": {"win":0,"podium":0,"top5":0,"n":0,"rows":[]}}

for birth in range(2015, 2023):
    for rt in ["derby","oaks"]:
        result = classic.get(rt,{}).get(str(birth), [])
        if len(result) < 5: continue
        winner = result[0]
        podium = set(result[:3])    # 1-3着
        top5 = set(result[:5])      # 1-5着
        horses = pd.read_csv(f"data/horses_{birth}.csv")
        sex = "牡" if rt=="derby" else "牝"
        feats = build_feature_matrix(horses, birth_year=birth, sex_filter=sex)
        feats["score"] = heuristic_score(feats, race_type=rt)
        pred5 = set(feats.sort_values("score",ascending=False).head(5)["horse_id"].astype(str))
        has_win = winner in pred5
        has_podium = bool(pred5 & podium)
        has_top5 = bool(pred5 & top5)
        a = agg[rt]
        a["n"] += 1
        if has_win: a["win"] += 1
        if has_podium: a["podium"] += 1
        if has_top5: a["top5"] += 1
        a["rows"].append((birth, "o" if has_win else "-", "o" if has_podium else "-", "o" if has_top5 else "-"))

print("=== 年度別 ===")
print("{:<6} {:<6} {:>4} {:>6} {:>6}".format("race","birth","1着","1-3着","1-5着"))
for rt in ["derby","oaks"]:
    for row in agg[rt]["rows"]:
        b, w, p, t = row
        print("{:<6} {:<6} {:>4} {:>6} {:>6}".format(rt, b, w, p, t))

print("\n=== 8年中の年数 ===")
print("{:<6} {:>4} {:>6} {:>6}".format("race","1着","1-3着","1-5着"))
for rt in ["derby","oaks"]:
    a = agg[rt]
    print("{:<6} {:>4} {:>6} {:>6}".format(rt, f"{a['win']}/{a['n']}", f"{a['podium']}/{a['n']}", f"{a['top5']}/{a['n']}"))

print("\n=== 率 (%) ===")
print("{:<6} {:>6} {:>6} {:>6}".format("race","1着","1-3着","1-5着"))
for rt in ["derby","oaks"]:
    a = agg[rt]
    n = a["n"]
    print("{:<6} {:>5.1f}% {:>5.1f}% {:>5.1f}%".format(rt, a["win"]/n*100, a["podium"]/n*100, a["top5"]/n*100))
