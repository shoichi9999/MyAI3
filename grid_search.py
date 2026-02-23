"""
グリッドサーチ — ヒューリスティックスコアの重み最適化。

使い方:
  python grid_search.py --years 2019
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features import build_feature_matrix


# ------------------------------------------------------------------
# パラメータ化されたスコア関数
# ------------------------------------------------------------------

def parameterized_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """パラメータ辞書でスコアを計算する。"""
    score = pd.Series(0.0, index=df.index)

    # 性別ボーナス
    if "sex" in df.columns:
        sex = df["sex"].fillna(0.5)
        score += (sex - 0.5) * params.get("b_sex", 10)

    w_sire = params["w_sire_ei"]
    w_dam = params["w_dam_prize"]
    w_bms = params["w_bms_ei"]

    # 父EI
    if "sire_ei" in df.columns:
        ei = df["sire_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * w_sire

    # 母父EI
    if "bms_ei" in df.columns:
        ei = df["bms_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * w_bms

    # 初年度種牡馬ボーナス（年内正規化）
    if "sire_prize" in df.columns and "sire_ei" in df.columns:
        is_first_crop = df["sire_ei"].fillna(0) == 0
        sire_prize_log = np.log1p(df["sire_prize"].fillna(0))
        fc_max = sire_prize_log[is_first_crop].max() if is_first_crop.any() else 0
        normalized = (sire_prize_log / fc_max) if fc_max > 0 else sire_prize_log * 0
        score += is_first_crop * normalized * params["w_first_crop"]

    # 母馬賞金
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        cap = dp.quantile(0.99)
        if cap > 0:
            score += (dp.clip(upper=cap) / cap) * 100 * w_dam

    # 調教師
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * params["w_trainer"]

    # 馬主
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * params["w_owner"]

    # 早生まれ
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * params["b_early"]

    # 両親若齢
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * params["b_parents_young"]
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * (params["b_parents_young"] / 2)

    # 母-母父年齢差
    if "dam_bms_gap_small" in df.columns:
        score += df["dam_bms_gap_small"].fillna(0) * params["b_dam_bms_gap"]

    # 種牡馬高齢ペナルティ
    if "sire_age" in df.columns:
        sa = df["sire_age"].fillna(12)
        score -= np.maximum(0, sa - 16) * params.get("b_sire_old", 0)

    # セリ価格
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * params["b_sale_price"]

    # 産駒番号
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -params["b_foal_penalty"],
                          np.where(fn <= 4, params["b_foal_bonus"], 0))

    # 生産牧場
    if "breeder_score" in df.columns:
        bs = df["breeder_score"].fillna(50)
        score += (bs - 50) * params["w_breeder"]

    # 母馬の繁殖入り年齢
    if "dam_breeding_age" in df.columns:
        dba = df["dam_breeding_age"]
        score += np.where(dba.isna(), 0,
                          (params["dam_breed_base"] - dba).clip(
                              -params["dam_breed_penalty"], params["dam_breed_cap"]))

    return score


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_single(df: pd.DataFrame, params: dict) -> dict:
    """1年度のデータで評価する。"""
    scores = parameterized_score(df, params).values
    y_true = df["prize_num"].values

    corr, _ = spearmanr(scores, y_true)
    if np.isnan(corr):
        corr = 0.0

    results = {}
    for n in [10, 30, 50, 100]:
        pred_idx = set(np.argsort(-scores)[:n])
        actual_idx = set(np.argsort(-y_true)[:n])
        results[f"top{n}"] = len(pred_idx & actual_idx)

    overall_avg = y_true.mean()
    pred_top30_avg = y_true[np.argsort(-scores)[:30]].mean()
    ratio = pred_top30_avg / overall_avg if overall_avg > 0 else 0

    return {
        "spearman": corr,
        "top10": results["top10"],
        "top30": results["top30"],
        "top50": results["top50"],
        "top100": results["top100"],
        "prize_ratio": ratio,
    }


def composite_score(metrics: dict, objective: str = "balanced") -> float:
    """複合スコア。objectiveで重み付けを切り替える。"""
    if objective == "top10":
        # TOP10を絶対的に優先。TOP30は同スコア時のタイブレーカーのみ
        return (
            metrics["top10"] * 100.0
            + metrics["top30"] * 1.0
            + metrics["spearman"] * 0.1
        )
    # balanced（従来）
    return (
        metrics["top30"] * 2.0
        + metrics["top10"] * 3.0
        + metrics["top100"] * 0.5
        + metrics["prize_ratio"] * 0.3
        + metrics["spearman"] * 10
    )


# グローバル設定（grid_search関数内で設定）
_OBJECTIVE = "balanced"


def cv_score(all_data: dict, params: dict) -> float:
    """全年度の composite_score 平均を返す。"""
    scores = []
    for df in all_data.values():
        m = evaluate_single(df, params)
        scores.append(composite_score(m, _OBJECTIVE))
    return np.mean(scores)


# ------------------------------------------------------------------
# データ読み込み
# ------------------------------------------------------------------

def _parse_prize(x) -> float:
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    return float(str(x).replace(",", "").replace("万", "").strip())


def load_year(year: int) -> pd.DataFrame | None:
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        return None
    horses = pd.read_csv(csv_path)
    horses["prize_num"] = horses["total_prize"].apply(_parse_prize)
    features = build_feature_matrix(horses, birth_year=year)
    features["prize_num"] = horses["prize_num"].values
    return features


# ------------------------------------------------------------------
# グリッドサーチ
# ------------------------------------------------------------------

def grid_search(years, objective="balanced"):
    global _OBJECTIVE
    _OBJECTIVE = objective
    print(f"=== データ読み込み === (目的関数: {objective})")
    all_data = {}
    for y in years:
        df = load_year(y)
        if df is not None:
            all_data[y] = df
            print(f"  {y}年: {len(df)}頭")
        else:
            print(f"  {y}年: データなし（スキップ）")

    if not all_data:
        print("[ERROR] データが見つかりません")
        return

    print(f"\n  利用年度: {sorted(all_data.keys())} ({len(all_data)}年分)")

    # 現行パラメータ（data/config/weights.json から読み込み）
    from src.model import _load_weights
    _w = _load_weights()
    current_params = {
        "b_sex": _w.get("b_sex", 16.47),
        "w_sire_ei": _w.get("w_sire_ei", 0.065),
        "w_dam_prize": _w.get("w_dam_prize", 0.036),
        "w_bms_ei": _w.get("w_bms_ei", 0.0235),
        "w_first_crop": _w.get("w_first_crop", 0.455),
        "b_early": _w.get("b_early", 0.0),
        "b_parents_young": _w.get("b_parents_young", 0.0),
        "b_dam_bms_gap": _w.get("b_dam_bms_gap", 0.0),
        "b_sale_price": _w.get("b_sale_price", 0.0),
        "b_foal_penalty": _w.get("b_foal_penalty", 14.97),
        "b_foal_bonus": _w.get("b_foal_bonus", 6.24),
        "w_trainer": _w.get("w_trainer", 0.270),
        "w_owner": _w.get("w_owner", 0.143),
        "w_breeder": _w.get("w_breeder", 0.0),
        "dam_breed_base": _w.get("dam_breed_base", 5.0),
        "dam_breed_cap": _w.get("dam_breed_cap", 2.0),
        "dam_breed_penalty": _w.get("dam_breed_penalty", 3.0),
        "b_sire_old": _w.get("b_sire_old", 0.0),
    }

    current_cv = cv_score(all_data, current_params)
    print(f"\n  現行パラメータのCVスコア: {current_cv:.2f}")
    for y, df in sorted(all_data.items()):
        m = evaluate_single(df, current_params)
        print(f"    {y}年: Spearman={m['spearman']:.4f} TOP10={m['top10']} TOP30={m['top30']} TOP100={m['top100']} 賞金倍率={m['prize_ratio']:.2f}x")

    best_score = current_cv
    best_params = dict(current_params)

    if objective == "top10":
        # TOP10最適化: 全パラメータ同時ランダム探索（局所最適回避）
        best_params, best_score = _random_search_top10(all_data, current_params, best_score)
    else:
        # balanced: 従来の段階的グリッドサーチ
        best_params, best_score = _staged_grid_search(all_data, dict(current_params), best_score)

    # ------------------------------------------------------------------
    # 最終結果
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  最適パラメータ")
    print("=" * 70)
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")
    print(f"\n  CVスコア: {best_score:.2f} (現行: {current_cv:.2f}, 差: {best_score - current_cv:+.2f})")

    # weights.json に書き戻し
    import json as _json
    weights_path = "data/config/weights.json"
    save_weights = dict(best_params)
    save_weights["_comment"] = "グリッドサーチ自動更新"
    with open(weights_path, "w", encoding="utf-8") as _f:
        _json.dump(save_weights, _f, ensure_ascii=False, indent=2)
    print(f"\n  → {weights_path} に保存しました")

    # 変更点
    print(f"\n--- 変更点 ---")
    changed = False
    for k in sorted(current_params):
        if current_params[k] != best_params[k]:
            print(f"  {k}: {current_params[k]} → {best_params[k]}")
            changed = True
    if not changed:
        print("  （変更なし — 現行パラメータが最適）")

    # 年度別詳細
    print(f"\n--- 年度別パフォーマンス比較 ---")
    print(f"{'年':>6} | {'指標':>10} | {'現行':>8} | {'最適化':>8} | {'差':>8}")
    print("-" * 60)
    for y in sorted(all_data.keys()):
        m_old = evaluate_single(all_data[y], current_params)
        m_new = evaluate_single(all_data[y], best_params)
        print(f"  {y} | {'Spearman':>10} | {m_old['spearman']:8.4f} | {m_new['spearman']:8.4f} | {m_new['spearman']-m_old['spearman']:+8.4f}")
        print(f"       | {'TOP10':>10} | {m_old['top10']:8d} | {m_new['top10']:8d} | {m_new['top10']-m_old['top10']:+8d}")
        print(f"       | {'TOP30':>10} | {m_old['top30']:8d} | {m_new['top30']:8d} | {m_new['top30']-m_old['top30']:+8d}")
        print(f"       | {'TOP100':>10} | {m_old['top100']:8d} | {m_new['top100']:8d} | {m_new['top100']-m_old['top100']:+8d}")
        print(f"       | {'賞金倍率':>10} | {m_old['prize_ratio']:8.2f}x | {m_new['prize_ratio']:8.2f}x | {m_new['prize_ratio']-m_old['prize_ratio']:+8.2f}")
        print("-" * 60)

    return best_params


def _precompute_arrays(all_data):
    """DataFrameから高速評価用のnumpy配列を事前計算する。"""
    precomputed = {}
    for year, df in all_data.items():
        n = len(df)
        d = {}
        # 性別
        d["sex_centered"] = (df["sex"].fillna(0.5) - 0.5).values if "sex" in df.columns else np.zeros(n)
        # 父EI（正規化済み）
        if "sire_ei" in df.columns:
            ei = df["sire_ei"].fillna(0).values
            cap = np.percentile(ei, 99)
            d["sire_ei_norm"] = (np.clip(ei, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_ei_norm"] = np.zeros(n)
        # 母父EI（正規化済み）
        if "bms_ei" in df.columns:
            ei = df["bms_ei"].fillna(0).values
            cap = np.percentile(ei, 99)
            d["bms_ei_norm"] = (np.clip(ei, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["bms_ei_norm"] = np.zeros(n)
        # 初年度種牡馬ボーナス（年内正規化）
        if "sire_prize" in df.columns and "sire_ei" in df.columns:
            is_first = (df["sire_ei"].fillna(0) == 0).values.astype(float)
            sp_log = np.log1p(df["sire_prize"].fillna(0).values)
            fc_mask = is_first.astype(bool)
            fc_max = sp_log[fc_mask].max() if fc_mask.any() else 1.0
            normalized = (sp_log / fc_max) if fc_max > 0 else np.zeros(n)
            d["first_crop_val"] = is_first * normalized
        else:
            d["first_crop_val"] = np.zeros(n)
        # 母馬賞金（対数正規化）
        if "dam_prize" in df.columns:
            dp = np.log1p(df["dam_prize"].fillna(0).values)
            cap = np.percentile(dp, 99)
            d["dam_prize_norm"] = (np.clip(dp, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["dam_prize_norm"] = np.zeros(n)
        # 調教師・馬主・牧場（中心化済み）
        d["trainer_centered"] = (df["trainer_score"].fillna(50) - 50).values if "trainer_score" in df.columns else np.zeros(n)
        d["owner_centered"] = (df["owner_score"].fillna(50) - 50).values if "owner_score" in df.columns else np.zeros(n)
        d["breeder_centered"] = (df["breeder_score"].fillna(50) - 50).values if "breeder_score" in df.columns else np.zeros(n)
        # 早生まれ
        d["early_born"] = df["early_born"].fillna(0).values if "early_born" in df.columns else np.zeros(n)
        # 両親若齢
        if "both_parents_young" in df.columns:
            d["parents_young"] = df["both_parents_young"].fillna(0).values
        elif "sire_young" in df.columns and "dam_young" in df.columns:
            d["parents_young_half"] = (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)).values
        else:
            d["parents_young"] = np.zeros(n)
        # 母-母父年齢差
        d["dam_bms_gap"] = df["dam_bms_gap_small"].fillna(0).values if "dam_bms_gap_small" in df.columns else np.zeros(n)
        # 種牡馬高齢ペナルティ（16歳超の超過年数）
        if "sire_age" in df.columns:
            sa = df["sire_age"].fillna(12).values
            d["sire_old_excess"] = np.maximum(0, sa - 16)
        else:
            d["sire_old_excess"] = np.zeros(n)
        # セリ価格（正規化）
        if "sale_price_log" in df.columns:
            sp = df["sale_price_log"].fillna(0).values
            max_sp = sp.max()
            d["sale_price_norm"] = (sp / max_sp) if max_sp > 0 else np.zeros(n)
        else:
            d["sale_price_norm"] = np.zeros(n)
        # 産駒番号（事前マスク）
        if "foal_number" in df.columns:
            fn = df["foal_number"].fillna(3).values
            d["is_first_foal"] = (fn == 1).astype(float)
            d["is_good_foal"] = ((fn >= 2) & (fn <= 4)).astype(float)
        else:
            d["is_first_foal"] = np.zeros(n)
            d["is_good_foal"] = np.zeros(n)
        # 繁殖入り年齢
        if "dam_breeding_age" in df.columns:
            dba = df["dam_breeding_age"].values
            d["dam_breed_notna"] = (~np.isnan(dba)).astype(float)
            d["dam_breed_age"] = np.nan_to_num(dba, nan=0.0)
        else:
            d["dam_breed_notna"] = np.zeros(n)
            d["dam_breed_age"] = np.zeros(n)
        # 正解データ
        d["prize"] = df["prize_num"].values
        d["actual_top10"] = set(np.argsort(-d["prize"])[:10])
        d["actual_top30"] = set(np.argsort(-d["prize"])[:30])

        precomputed[year] = d
    return precomputed


def _fast_cv_score(precomputed, params):
    """事前計算配列を使った高速CVスコア。"""
    total_score = 0.0
    n_years = len(precomputed)

    for d in precomputed.values():
        # ベクトル化されたスコア計算
        score = (
            d["sex_centered"] * params[0]          # b_sex
            + d["sire_ei_norm"] * params[1]         # w_sire_ei
            + d["dam_prize_norm"] * params[2]       # w_dam_prize
            + d["bms_ei_norm"] * params[3]          # w_bms_ei
            + d["first_crop_val"] * params[4]       # w_first_crop
            + d["trainer_centered"] * params[5]     # w_trainer
            + d["owner_centered"] * params[6]       # w_owner
            + d["breeder_centered"] * params[7]     # w_breeder
            + d["early_born"] * params[8]           # b_early
            + d["dam_bms_gap"] * params[10]         # b_dam_bms_gap
            + d["sale_price_norm"] * params[11]     # b_sale_price
            + d["is_first_foal"] * (-params[12])    # b_foal_penalty
            + d["is_good_foal"] * params[13]        # b_foal_bonus
            - d["sire_old_excess"] * params[17]     # b_sire_old
        )
        # 両親若齢（2パターン）
        if "parents_young" in d:
            score += d["parents_young"] * params[9]
        else:
            score += d["parents_young_half"] * (params[9] / 2)

        # 繁殖入り年齢（clip付き）
        breed_val = np.clip(params[14] - d["dam_breed_age"], -params[16], params[15])
        score += d["dam_breed_notna"] * breed_val

        # TOP10/TOP30 計算
        pred_top10 = set(np.argpartition(-score, 10)[:10])
        pred_top30 = set(np.argpartition(-score, 30)[:30])

        top10_match = len(pred_top10 & d["actual_top10"])
        top30_match = len(pred_top30 & d["actual_top30"])

        # Spearmanは高コストなので簡易版: TOP10支配なので省略可能
        # composite: top10 * 100 + top30 * 1 + spearman * 0.1
        total_score += top10_match * 100.0 + top30_match * 1.0

    return total_score / n_years


# パラメータ名 → 配列インデックスの対応
_PARAM_KEYS = [
    "b_sex", "w_sire_ei", "w_dam_prize", "w_bms_ei", "w_first_crop",
    "w_trainer", "w_owner", "w_breeder", "b_early", "b_parents_young",
    "b_dam_bms_gap", "b_sale_price", "b_foal_penalty", "b_foal_bonus",
    "dam_breed_base", "dam_breed_cap", "dam_breed_penalty",
    "b_sire_old",
]

def _dict_to_arr(params):
    return np.array([params[k] for k in _PARAM_KEYS])

def _arr_to_dict(arr):
    return {k: float(v) for k, v in zip(_PARAM_KEYS, arr)}


def _random_search_top10(all_data, current_params, best_score):
    """マルチリスタート+ポピュレーションベース探索（TOP10最大化専用、高速版）。"""
    import time

    # 事前計算（1回だけ）
    print("\n  特徴量を事前計算中...")
    precomputed = _precompute_arrays(all_data)
    print("  完了")

    # パラメータの探索範囲
    param_ranges = np.array([
        (0, 20),       # b_sex
        (0.0, 0.50),   # w_sire_ei
        (0.0, 0.30),   # w_dam_prize
        (0.0, 0.35),   # w_bms_ei
        (0.0, 25.0),   # w_first_crop (正規化後は[0,1]なのでpt単位)
        (0.0, 0.50),   # w_trainer
        (0.0, 0.50),   # w_owner
        (0.0, 0.50),   # w_breeder
        (0, 15),       # b_early
        (0, 15),       # b_parents_young
        (0, 15),       # b_dam_bms_gap
        (0, 15),       # b_sale_price
        (0, 15),       # b_foal_penalty
        (0, 10),       # b_foal_bonus
        (1, 12),       # dam_breed_base
        (0, 8),        # dam_breed_cap
        (0, 8),        # dam_breed_penalty
        (0, 5),        # b_sire_old
    ])
    lo = param_ranges[:, 0]
    hi = param_ranges[:, 1]
    span = hi - lo
    n_params = len(lo)

    current_arr = _dict_to_arr(current_params)
    best_arr = current_arr.copy()
    # 事前計算版の現行スコア
    best_score_fast = _fast_cv_score(precomputed, current_arr)
    print(f"  現行スコア(高速版): {best_score_fast:.2f}")

    def _show_top10(arr, label=""):
        params = _arr_to_dict(arr)
        top10s = []
        total = 0
        for y in sorted(all_data.keys()):
            m = evaluate_single(all_data[y], params)
            top10s.append(f"{y}:{m['top10']}")
            total += m['top10']
        print(f"  {label} TOP10合計={total} [{', '.join(top10s)}]")

    t_start = time.time()

    # ================================================================
    # Phase 1: マルチシード広域探索 (5シード × 100k = 500k)
    # ================================================================
    n_seeds = 5
    n_phase1 = 100000
    print(f"\n{'='*60}")
    print(f"  Phase 1: マルチシード広域探索 ({n_seeds}シード × {n_phase1:,} = {n_seeds*n_phase1:,}回)")
    print(f"{'='*60}")

    pool_size = 50
    pool = [(best_score_fast, best_arr.copy())]

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 1000 + 42)
        local_best = best_score_fast
        local_arr = best_arr.copy()

        for i in range(n_phase1):
            p = rng.uniform(lo, hi)
            s = _fast_cv_score(precomputed, p)
            if s > local_best:
                local_best = s
                local_arr = p.copy()
            if s > pool[-1][0] if len(pool) >= pool_size else True:
                pool.append((s, p.copy()))

        pool.sort(key=lambda x: -x[0])
        pool = pool[:pool_size]

        elapsed = time.time() - t_start
        _show_top10(local_arr, f"Seed {seed} ({elapsed:.0f}s): score={local_best:.2f}")

    best_score_fast, best_arr = pool[0]
    print(f"\n  Phase 1 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
    _show_top10(best_arr, "Best")

    # ================================================================
    # Phase 2: トップ候補の局所探索 (上位30個 × 20k = 600k)
    # ================================================================
    n_top = 30
    n_phase2 = 20000
    print(f"\n{'='*60}")
    print(f"  Phase 2: トップ{n_top}局所探索 ({n_top} × {n_phase2:,} = {n_top*n_phase2:,}回)")
    print(f"{'='*60}")

    rng = np.random.default_rng(9999)
    new_pool = list(pool[:pool_size])

    for ci in range(min(n_top, len(pool))):
        cand_score, cand_arr = pool[ci]
        local_best = cand_score
        local_arr = cand_arr.copy()

        for i in range(n_phase2):
            delta = rng.uniform(-0.15, 0.15, n_params) * span
            p = np.clip(local_arr + delta, lo, hi)
            s = _fast_cv_score(precomputed, p)
            if s > local_best:
                local_best = s
                local_arr = p.copy()

        new_pool.append((local_best, local_arr.copy()))
        if ci < 5:
            _show_top10(local_arr, f"Cand {ci}: score={local_best:.2f}")

    new_pool.sort(key=lambda x: -x[0])
    new_pool = new_pool[:pool_size]
    best_score_fast, best_arr = new_pool[0]
    print(f"\n  Phase 2 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
    _show_top10(best_arr, "Best")

    # ================================================================
    # Phase 3: トップ10の微調整 (10 × 30k = 300k)
    # ================================================================
    n_fine = 10
    n_phase3 = 30000
    print(f"\n{'='*60}")
    print(f"  Phase 3: トップ{n_fine}微調整 ({n_fine} × {n_phase3:,} = {n_fine*n_phase3:,}回)")
    print(f"{'='*60}")

    for ci in range(min(n_fine, len(new_pool))):
        cand_score, cand_arr = new_pool[ci]
        local_best = cand_score
        local_arr = cand_arr.copy()

        for i in range(n_phase3):
            delta = rng.uniform(-0.05, 0.05, n_params) * span
            p = np.clip(local_arr + delta, lo, hi)
            s = _fast_cv_score(precomputed, p)
            if s > local_best:
                local_best = s
                local_arr = p.copy()

        if ci < 5:
            _show_top10(local_arr, f"Fine {ci}: score={local_best:.2f}")
        if local_best > best_score_fast:
            best_score_fast = local_best
            best_arr = local_arr.copy()

    print(f"\n  Phase 3 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
    _show_top10(best_arr, "Best")

    # ================================================================
    # Phase 4: 最終超微調整 (100k at ±2%)
    # ================================================================
    n_phase4 = 100000
    print(f"\n{'='*60}")
    print(f"  Phase 4: 最終超微調整 ({n_phase4:,}回 at ±2%)")
    print(f"{'='*60}")

    for i in range(n_phase4):
        delta = rng.uniform(-0.02, 0.02, n_params) * span
        p = np.clip(best_arr + delta, lo, hi)
        s = _fast_cv_score(precomputed, p)
        if s > best_score_fast:
            best_score_fast = s
            best_arr = p.copy()

    elapsed = time.time() - t_start
    print(f"  Phase 4 完了: best={best_score_fast:.2f} ({elapsed:.0f}s)")
    _show_top10(best_arr, "Final")

    # dict形式で返す（元のcv_scoreで検算）
    best_params = _arr_to_dict(best_arr)
    best_score = cv_score(all_data, best_params)
    print(f"\n  検算(元スコア関数): {best_score:.2f}")
    print(f"  総所要時間: {elapsed:.0f}秒")

    return best_params, best_score


def _staged_grid_search(all_data, best_params, best_score):
    """従来の段階的グリッドサーチ（balanced用）。"""
    # ===============================================================
    # Stage 1: 血統重み（父EI + 母父EI + 母馬賞金 + 初年度ボーナス）
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 1: 血統重み最適化")
    print(f"{'='*60}")
    blood_combos = list(itertools.product(
        [0.15, 0.20, 0.225, 0.25, 0.30, 0.35],   # w_sire_ei
        [0.05, 0.075, 0.10, 0.15, 0.20],           # w_dam_prize
        [0.10, 0.15, 0.20, 0.25, 0.30],            # w_bms_ei
        [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],            # w_first_crop
    ))
    print(f"  組み合わせ数: {len(blood_combos)}")

    for ws, wd, wb, wf in blood_combos:
        p = dict(best_params)
        p["w_sire_ei"] = ws
        p["w_dam_prize"] = wd
        p["w_bms_ei"] = wb
        p["w_first_crop"] = wf
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"w_sire_ei": ws, "w_dam_prize": wd,
                                "w_bms_ei": wb, "w_first_crop": wf})

    print(f"  最良: sire_ei={best_params['w_sire_ei']}, dam_prize={best_params['w_dam_prize']}, "
          f"bms_ei={best_params['w_bms_ei']}, first_crop={best_params['w_first_crop']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 2: ボーナスパラメータ
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 2: ボーナス最適化")
    print(f"{'='*60}")
    bonus_combos = list(itertools.product(
        [0, 3, 5, 8, 10, 12],     # b_early
        [0, 3, 5, 8, 10, 12],     # b_parents_young
        [0, 2, 5, 8],             # b_dam_bms_gap
        [0, 3, 5, 8, 10],         # b_sale_price
    ))
    print(f"  組み合わせ数: {len(bonus_combos)}")

    for be, bp, bg, bs in bonus_combos:
        p = dict(best_params)
        p["b_early"] = be
        p["b_parents_young"] = bp
        p["b_dam_bms_gap"] = bg
        p["b_sale_price"] = bs
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"b_early": be, "b_parents_young": bp,
                                "b_dam_bms_gap": bg, "b_sale_price": bs})

    print(f"  最良: early={best_params['b_early']}, parents_young={best_params['b_parents_young']}, "
          f"dam_bms_gap={best_params['b_dam_bms_gap']}, sale_price={best_params['b_sale_price']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 3: コネクション重み + 産駒番号
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 3: コネクション + 産駒番号")
    print(f"{'='*60}")
    conn_combos = list(itertools.product(
        [0, 2, 3, 5, 7],              # b_foal_penalty
        [0, 1, 2, 3, 5],              # b_foal_bonus
        [0.03, 0.05, 0.08, 0.12],     # w_trainer
        [0.03, 0.05, 0.08, 0.12],     # w_owner
        [0.05, 0.10, 0.15, 0.20],     # w_breeder
    ))
    print(f"  組み合わせ数: {len(conn_combos)}")

    for fp, fb, wt, wo, wb in conn_combos:
        p = dict(best_params)
        p["b_foal_penalty"] = fp
        p["b_foal_bonus"] = fb
        p["w_trainer"] = wt
        p["w_owner"] = wo
        p["w_breeder"] = wb
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"b_foal_penalty": fp, "b_foal_bonus": fb,
                                "w_trainer": wt, "w_owner": wo, "w_breeder": wb})

    print(f"  最良: foal_penalty={best_params['b_foal_penalty']}, foal_bonus={best_params['b_foal_bonus']}, "
          f"trainer={best_params['w_trainer']}, owner={best_params['w_owner']}, breeder={best_params['w_breeder']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 4: 繁殖入り年齢パラメータ
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 4: 繁殖入り年齢パラメータ")
    print(f"{'='*60}")
    breed_combos = list(itertools.product(
        [5, 6, 7, 8],     # dam_breed_base
        [2, 3, 4, 5],     # dam_breed_cap
        [1, 2, 3],        # dam_breed_penalty
    ))
    print(f"  組み合わせ数: {len(breed_combos)}")

    for base, cap, pen in breed_combos:
        p = dict(best_params)
        p["dam_breed_base"] = base
        p["dam_breed_cap"] = cap
        p["dam_breed_penalty"] = pen
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"dam_breed_base": base, "dam_breed_cap": cap,
                                "dam_breed_penalty": pen})

    print(f"  最良: base={best_params['dam_breed_base']}, cap={best_params['dam_breed_cap']}, "
          f"penalty={best_params['dam_breed_penalty']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 5: 血統重み微調整
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 5: 血統重み微調整")
    print(f"{'='*60}")
    def _fine(val, step=0.025):
        return sorted(set([val + d for d in [-step, -step/2, 0, step/2, step]]))

    fine_combos = list(itertools.product(
        _fine(best_params["w_sire_ei"]),
        _fine(best_params["w_dam_prize"]),
        _fine(best_params["w_bms_ei"]),
        _fine(best_params["w_first_crop"], step=0.1),
    ))
    print(f"  組み合わせ数: {len(fine_combos)}")

    for ws, wd, wb, wf in fine_combos:
        if ws <= 0 or wd <= 0 or wb <= 0 or wf <= 0:
            continue
        p = dict(best_params)
        p["w_sire_ei"] = ws
        p["w_dam_prize"] = wd
        p["w_bms_ei"] = wb
        p["w_first_crop"] = wf
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"w_sire_ei": ws, "w_dam_prize": wd,
                                "w_bms_ei": wb, "w_first_crop": wf})

    return best_params, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="グリッドサーチ（重み最適化）")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                        help="使用する年度リスト")
    parser.add_argument("--objective", choices=["balanced", "top10"],
                        default="balanced",
                        help="最適化目的: balanced(従来) / top10(TOP10最大化)")
    args = parser.parse_args()

    best = grid_search(args.years, objective=args.objective)
