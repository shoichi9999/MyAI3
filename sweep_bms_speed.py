"""w_mid_x_speed (オークス) のスイープ。"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.features import build_feature_matrix
from src.model import _load_weights, heuristic_score

classic = json.load(open("data/classic_results.json", encoding="utf-8"))

def eval_year(birth, w_override):
    horses = pd.read_csv(f"data/horses_{birth}.csv")
    feats = build_feature_matrix(horses, birth_year=birth, sex_filter="牝")
    base_score = heuristic_score(feats, race_type="oaks")
    cur_w = _load_weights("oaks").get("w_mid_x_speed", 0.0)
    adj = (w_override - cur_w) * feats["mid_x_speed"].fillna(0)
    score = base_score + adj
    feats["score"] = score
    feats = feats.sort_values("score", ascending=False).reset_index(drop=True)
    feats["rank"] = feats.index + 1

    top5_ids = classic.get("oaks", {}).get(str(birth), [])
    if not top5_ids: return None
    winner = top5_ids[0]
    pred_top5 = set(feats.head(5)["horse_id"].astype(str))
    w_rank = None
    row = feats[feats["horse_id"].astype(str) == winner]
    if not row.empty:
        w_rank = int(row.iloc[0]["rank"])
    return {"winner_rank": w_rank, "winner_in_top5": winner in pred_top5}

print("=== w_mid_x_speed スイープ (オークス 2015-2023) ===\n")
print(f"{'w':>6} | " + " ".join(f"{y:>5}" for y in range(2015,2024)) + " | TOP1 TOP5_winner avg")
print("-" * 100)
for w in [0, 5, 10, 20, 30, 50, 80, 120]:
    ranks = []; in5 = 0; t1 = 0; cells = []
    for birth in range(2015, 2024):
        m = eval_year(birth, w)
        if m is None: cells.append("-"); continue
        ranks.append(m["winner_rank"])
        if m["winner_in_top5"]: in5 += 1
        if m["winner_rank"] == 1: t1 += 1
        cells.append(f"{m['winner_rank']:>5}")
    avg = np.mean(ranks) if ranks else 0
    print(f"{w:>6} | " + " ".join(cells) + f" | {t1}/9   {in5}/9      {avg:.1f}")
