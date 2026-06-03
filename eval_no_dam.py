"""母馬直接実績を排除した場合の評価。

w_dam_prize, b_dam_high_class, w_sire_dam_inter, w_bms_dam_inter を 0 にして
バックテスト + 2024年産TOP10予測を表示。
weights.json は触らない（一時的な monkey patch）。
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

import src.model as model_mod
_orig = model_mod._load_weights

ZERO_KEYS = ["w_dam_prize", "b_dam_high_class", "w_sire_dam_inter", "w_bms_dam_inter"]

def patched_load(race_type):
    w = dict(_orig(race_type))
    for k in ZERO_KEYS:
        w[k] = 0.0
    return w

model_mod._load_weights = patched_load

from src.features import build_feature_matrix
from src.model import heuristic_score

classic = json.load(open("data/classic_results.json", encoding="utf-8"))

def eval_year(birth, race_type):
    horses = pd.read_csv(f"data/horses_{birth}.csv")
    sex = "牡" if race_type == "derby" else "牝"
    feats = build_feature_matrix(horses, birth_year=birth, sex_filter=sex)
    feats["score"] = heuristic_score(feats, race_type=race_type)
    feats = feats.sort_values("score", ascending=False).reset_index(drop=True)
    feats["rank"] = feats.index + 1
    top5_ids = classic.get(race_type, {}).get(str(birth), [])
    if not top5_ids: return None
    winner = top5_ids[0]
    pred_top5 = set(feats.head(5)["horse_id"].astype(str))
    row = feats[feats["horse_id"].astype(str) == winner]
    w_rank = int(row.iloc[0]["rank"]) if not row.empty else None
    pred_top5_hits = len(pred_top5 & set(top5_ids))
    return {"winner_rank": w_rank, "winner_in_top5": winner in pred_top5,
            "top5_hits": pred_top5_hits}

# バックテスト
print("=== 母馬実績抜き (A) のバックテスト ===\n")
for race in ["derby", "oaks"]:
    ranks = []; t1 = 0; in_top5 = 0; top5_total = 0; cells = []
    label = "ダービー" if race == "derby" else "オークス"
    print(f"--- {label} ---")
    print(f'{"生年":>4} {"1着順位":>8} {"TOP5入り":>8} {"TOP5的中":>8}')
    for birth in range(2015, 2024):
        m = eval_year(birth, race)
        if m is None: continue
        ranks.append(m["winner_rank"])
        if m["winner_in_top5"]: in_top5 += 1
        if m["winner_rank"] == 1: t1 += 1
        top5_total += m["top5_hits"]
        print(f'  {birth:>4} {m["winner_rank"]:>8} {"○" if m["winner_in_top5"] else "-":>8} {m["top5_hits"]:>8}/5')
    print(f'  TOP1的中: {t1}/9, TOP5に1着含む: {in_top5}/9, TOP5合計的中: {top5_total}/45 (平均{top5_total/9:.2f}/5)')
    print(f'  1着平均順位: {np.mean(ranks):.1f}\n')

# 2024年産予測
horses_2024 = pd.read_csv("data/horses_2024.csv")
for race in ["derby", "oaks"]:
    sex = "牡" if race == "derby" else "牝"
    feats = build_feature_matrix(horses_2024, birth_year=2024, sex_filter=sex)
    feats["score"] = heuristic_score(feats, race_type=race)
    top = feats.sort_values("score", ascending=False).head(10)
    label = "ダービー" if race == "derby" else "オークス"
    print(f"=== 2024年産 {label} TOP10 (母馬実績抜き) ===")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f'  {i:>2}. {r["horse_name"]:<26} score={r["score"]:>7.1f}')
    print()
