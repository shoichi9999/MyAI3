"""新特徴量パラメータの高速チューニング。

既存の最適重みを固定し、新パラメータ5個のみを集中探索。
"""

import json
import time

import numpy as np

from grid_search import (
    _PARAM_KEYS, _dict_to_arr, _arr_to_dict,
    _precompute_arrays, _fast_cv_score, _get_classic_ids,
    load_year, evaluate_single, cv_score,
)
from src.model import _load_weights

# ---- データ読み込み ----
print("=== データ読み込み ===")
all_data = {}
for y in range(2015, 2023):
    df = load_year(y)
    if df is not None:
        all_data[y] = df
        print(f"  {y}年: {len(df)}頭")

# ---- 現行パラメータ ----
_w = _load_weights()
current_params = {k: _w.get(k, 0.0) for k in _PARAM_KEYS}
# 新パラメータのデフォルト
for k in ["w_bms_rank", "w_bms_progeny_prize", "w_sire_classic_rate", "w_owner_trainer", "w_bms_dam_inter"]:
    current_params.setdefault(k, 0.0)

current_cv = cv_score(all_data, current_params)
print(f"\n現行CVスコア: {current_cv:.2f}")
for y, df in sorted(all_data.items()):
    m = evaluate_single(df, current_params, birth_year=y)
    t10 = m.get("top10_total", 0)
    md10 = m.get("male_top10_derby", 0)
    fo10 = m.get("female_top10_oaks", 0)
    print(f"  {y}: TOP10={t10}/10 D{md10} O{fo10}")

# ---- 事前計算 ----
print("\n事前計算中...")
precomputed = _precompute_arrays(all_data)
current_arr = _dict_to_arr(current_params)
current_fast = _fast_cv_score(precomputed, current_arr)
print(f"現行スコア(高速): {current_fast:.2f}")

# ---- 新パラメータのインデックス ----
NEW_KEYS = ["w_bms_rank", "w_bms_progeny_prize", "w_sire_classic_rate", "w_owner_trainer", "w_bms_dam_inter"]
new_indices = [_PARAM_KEYS.index(k) for k in NEW_KEYS]
print(f"新パラメータ: {NEW_KEYS} (indices: {new_indices})")

# 探索範囲（新パラメータのみ）
new_bounds = {
    "w_bms_rank": (0.0, 0.60),
    "w_bms_progeny_prize": (0.0, 0.50),
    "w_sire_classic_rate": (0.0, 20.0),
    "w_owner_trainer": (0.0, 0.20),
    "w_bms_dam_inter": (0.0, 25.0),
}

# ---- Phase 1: 全既存パラメータも含めた探索 ----
# 既存パラメータは±10%の範囲で微調整、新パラメータは全範囲探索
print(f"\n{'='*60}")
print("  Phase 1: 新パラメータ + 既存パラメータ微調整 (500k回)")
print(f"{'='*60}")
t_start = time.time()

best_arr = current_arr.copy()
best_fast = current_fast
rng = np.random.default_rng(42)

lo = np.array([new_bounds[k][0] for k in NEW_KEYS])
hi = np.array([new_bounds[k][1] for k in NEW_KEYS])

n_total = len(_PARAM_KEYS)
full_lo = _dict_to_arr({k: max(0, current_params[k] * 0.7) for k in _PARAM_KEYS})
full_hi = _dict_to_arr({k: current_params[k] * 1.3 + 1.0 for k in _PARAM_KEYS})
# 新パラメータは全範囲
for i, k in zip(new_indices, NEW_KEYS):
    full_lo[i] = new_bounds[k][0]
    full_hi[i] = new_bounds[k][1]
# 一部の既存パラメータのバウンドを広げる
full_lo = np.maximum(full_lo, 0)

improved = 0
for iteration in range(500000):
    trial = best_arr.copy()
    # 新パラメータ: 全範囲ランダム or ローカル探索
    if rng.random() < 0.3:
        # 全範囲探索
        for i, k in zip(new_indices, NEW_KEYS):
            trial[i] = rng.uniform(new_bounds[k][0], new_bounds[k][1])
    else:
        # 現ベストの近傍
        delta = rng.uniform(-0.15, 0.15, n_total) * (full_hi - full_lo)
        trial = np.clip(best_arr + delta, full_lo, full_hi)

    s = _fast_cv_score(precomputed, trial)
    if s > best_fast:
        best_fast = s
        best_arr = trial.copy()
        improved += 1

    if (iteration + 1) % 100000 == 0:
        elapsed = time.time() - t_start
        print(f"  {iteration+1:>7,}回: best={best_fast:.2f} (改善{improved}回, {elapsed:.0f}s)")

# ---- Phase 2: 微調整 (±5%, ±2%) ----
print(f"\n{'='*60}")
print("  Phase 2: 微調整")
print(f"{'='*60}")
span = full_hi - full_lo

for step_size, n_iter, label in [(0.05, 300000, "±5%"), (0.02, 200000, "±2%")]:
    imp = 0
    for _ in range(n_iter):
        delta = rng.uniform(-step_size, step_size, n_total) * span
        trial = np.clip(best_arr + delta, full_lo, full_hi)
        s = _fast_cv_score(precomputed, trial)
        if s > best_fast:
            best_fast = s
            best_arr = trial.copy()
            imp += 1
    elapsed = time.time() - t_start
    print(f"  {label}: {best_fast:.2f} (改善{imp}回, {elapsed:.0f}s)")

# ---- 結果表示 ----
best_params = _arr_to_dict(best_arr)
best_cv = cv_score(all_data, best_params)

print(f"\n{'='*60}")
print("  最終結果")
print(f"{'='*60}")
print(f"  現行CVスコア: {current_cv:.2f}")
print(f"  最適CVスコア: {best_cv:.2f} (差: {best_cv - current_cv:+.2f})")

# 年度別
for y in sorted(all_data.keys()):
    m_old = evaluate_single(all_data[y], current_params, birth_year=y)
    m_new = evaluate_single(all_data[y], best_params, birth_year=y)
    t10_o = m_old.get("top10_total", 0)
    t10_n = m_new.get("top10_total", 0)
    md_o = m_old.get("male_top10_derby", 0)
    md_n = m_new.get("male_top10_derby", 0)
    fo_o = m_old.get("female_top10_oaks", 0)
    fo_n = m_new.get("female_top10_oaks", 0)
    print(f"  {y}: TOP10 {t10_o}→{t10_n}  D {md_o}→{md_n}  O {fo_o}→{fo_n}")

# 変更点
print("\n--- 変更点 ---")
for k in _PARAM_KEYS:
    old = current_params.get(k, 0)
    new = best_params.get(k, 0)
    if abs(old - new) > 0.001:
        print(f"  {k}: {old:.4f} → {new:.4f}")

# 保存
if best_cv > current_cv:
    save_weights = dict(best_params)
    save_weights["_comment"] = "グリッドサーチ自動更新"
    with open("data/config/weights.json", "w", encoding="utf-8") as f:
        json.dump(save_weights, f, ensure_ascii=False, indent=2)
    print(f"\n  → weights.json に保存しました")
else:
    print(f"\n  改善なし — weights.json は更新しません")
