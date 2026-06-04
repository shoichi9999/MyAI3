"""w_stayer_miler の値を変えてバックテスト結果を比較する。

ダービー側のみ評価（理論は牡馬ダービー向け）。
オークスは別途同じ理論が当てはまるか後で検証。
"""
import sys, json, copy
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.features import build_feature_matrix
from src.model import _load_weights, heuristic_score

classic = json.load(open("data/classic_results.json", encoding="utf-8"))

def eval_year(birth, race_type, w_override):
    """指定生年・レースでwを上書きしてスコア計算→1着順位/TOP5的中を返す。"""
    horses = pd.read_csv(f"data/horses_{birth}.csv")
    sex = "牡" if race_type == "derby" else "牝"
    feats = build_feature_matrix(horses, birth_year=birth, sex_filter=sex)

    # heuristic_scoreはweights.jsonをロードするので、一時的に書き換える代わりに自前計算
    # 既存scoreにstayer_x_miler項を足す
    base_score = heuristic_score(feats, race_type=race_type)
    # stayer_x_milerの寄与をw_overrideで差し替え
    cur_w = _load_weights(race_type).get("w_stayer_miler", 0.0)
    adj = (w_override - cur_w) * feats["stayer_x_miler"].fillna(0)
    score = base_score + adj

    feats["score"] = score
    feats = feats.sort_values("score", ascending=False).reset_index(drop=True)
    feats["rank"] = feats.index + 1

    top5_ids = classic.get(race_type, {}).get(str(birth), [])
    if not top5_ids: return None
    winner = top5_ids[0]
    pred_top5 = set(feats.head(5)["horse_id"].astype(str))

    w_rank = None
    row = feats[feats["horse_id"].astype(str) == winner]
    if not row.empty:
        w_rank = int(row.iloc[0]["rank"])
    hits_top5 = len(pred_top5 & set(top5_ids))
    return {"winner_rank": w_rank, "top5_hits": hits_top5, "winner_in_top5": winner in pred_top5}

print("=== w_stayer_miler スイープ (ダービー 2015-2023) ===\n")
print(f"{'w':>6} | " + " ".join(f"{y:>5}" for y in range(2015,2024)) + " | TOP1 TOP5_winner avg_rank")
print("-" * 100)
for w in [0, 30, 80, 120, 200, 300, 500, 1000, 2000]:
    ranks = []
    in_top5 = 0
    is_top1 = 0
    yearly = []
    for birth in range(2015, 2024):
        m = eval_year(birth, "derby", w)
        if m is None:
            yearly.append("-"); continue
        ranks.append(m["winner_rank"])
        if m["winner_in_top5"]: in_top5 += 1
        if m["winner_rank"] == 1: is_top1 += 1
        yearly.append(f"{m['winner_rank']:>5}")
    avg = np.mean(ranks) if ranks else 0
    print(f"{w:>6} | " + " ".join(yearly) + f" | {is_top1}/9   {in_top5}/9      {avg:.1f}")
