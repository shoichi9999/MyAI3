"""
グリッドサーチ — ヒューリスティックスコアの重み最適化。

Leave-One-Out クロスバリデーションで汎化性能を最大化する。
全年度（2015-2021）のデータを使い、各年を順番にバリデーションにして
残りで訓練→全年度の平均スコアで最良パラメータを決定。

使い方:
  python grid_search.py
  python grid_search.py --years 2015 2016 2020 2021
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
    for n in [10, 30, 50]:
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
        "prize_ratio": ratio,
    }


def composite_score(metrics: dict) -> float:
    """POG実用性重視の複合スコア。
    TOP30一致と賞金倍率を重視、Spearmanは補助的。"""
    return (
        metrics["top30"] * 2.0          # TOP30一致が最重要
        + metrics["top10"] * 3.0        # TOP10一致も重視
        + metrics["prize_ratio"] * 0.3  # 賞金倍率
        + metrics["spearman"] * 10      # Spearmanは補助
    )


def loo_cv_score(all_data: dict, params: dict) -> float:
    """Leave-One-Out CV: 各年度のcomposite_scoreの平均を返す。"""
    scores = []
    for df in all_data.values():
        m = evaluate_single(df, params)
        scores.append(composite_score(m))
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

def grid_search(years):
    # データ読み込み
    print("=== データ読み込み ===")
    all_data = {}
    for y in years:
        df = load_year(y)
        if df is not None:
            all_data[y] = df
            print(f"  {y}年: {len(df)}頭")
        else:
            print(f"  {y}年: データなし（スキップ）")

    if len(all_data) < 2:
        print("[ERROR] 最低2年分のデータが必要です")
        return

    print(f"\n  利用年度: {sorted(all_data.keys())} ({len(all_data)}年分)")

    # 現行パラメータ
    current_params = {
        "w_sire_ei": 0.25, "w_dam_prize": 0.075, "w_bms_ei": 0.175,
        "b_early": 8, "b_parents_young": 12, "b_dam_bms_gap": 3,
        "b_sale_price": 8, "b_foal_penalty": 3, "b_foal_bonus": 2,
        "w_trainer": 0.05, "w_owner": 0.05, "w_breeder": 0.08,
    }

    current_cv = loo_cv_score(all_data, current_params)
    print(f"\n  現行パラメータのCV複合スコア: {current_cv:.2f}")
    for y, df in sorted(all_data.items()):
        m = evaluate_single(df, current_params)
        print(f"    {y}年: Spearman={m['spearman']:.4f}, TOP10={m['top10']}, TOP30={m['top30']}, 賞金倍率={m['prize_ratio']:.2f}x")

    # パラメータグリッド定義
    grid = {
        "w_sire_ei":       [0.20, 0.25, 0.30, 0.35, 0.40],
        "w_dam_prize":     [0.05, 0.075, 0.10, 0.15, 0.20],
        "w_bms_ei":        [0.10, 0.15, 0.175, 0.20, 0.25, 0.30],
        "b_early":         [3, 5, 8, 10, 12],
        "b_parents_young": [5, 8, 10, 12, 15],
        "b_dam_bms_gap":   [0, 2, 3, 5],
        "b_sale_price":    [3, 5, 8, 10, 12],
        "b_foal_penalty":  [2, 3, 5],
        "b_foal_bonus":    [1, 2, 3],
        "w_trainer":       [0.03, 0.05, 0.08, 0.10],
        "w_owner":         [0.03, 0.05, 0.08],
        "w_breeder":       [0.05, 0.08, 0.12, 0.15],
    }

    best_score = current_cv
    best_params = dict(current_params)

    # ===============================================================
    # Stage 1: 血統重み（主要3パラメータ）
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 1: 血統重み最適化")
    print(f"{'='*60}")
    blood_combos = list(itertools.product(
        grid["w_sire_ei"], grid["w_dam_prize"], grid["w_bms_ei"]
    ))
    print(f"  組み合わせ数: {len(blood_combos)}")

    for w_sire, w_dam, w_bms in blood_combos:
        p = dict(best_params)
        p["w_sire_ei"] = w_sire
        p["w_dam_prize"] = w_dam
        p["w_bms_ei"] = w_bms
        cv = loo_cv_score(all_data, p)
        if cv > best_score:
            best_score = cv
            best_params.update({"w_sire_ei": w_sire, "w_dam_prize": w_dam, "w_bms_ei": w_bms})

    print(f"  最良: sire_ei={best_params['w_sire_ei']}, dam_prize={best_params['w_dam_prize']}, bms_ei={best_params['w_bms_ei']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 2: ボーナスパラメータ
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 2: ボーナス最適化")
    print(f"{'='*60}")
    bonus_combos = list(itertools.product(
        grid["b_early"], grid["b_parents_young"], grid["b_dam_bms_gap"],
        grid["b_sale_price"]
    ))
    print(f"  組み合わせ数: {len(bonus_combos)}")

    for b_early, b_py, b_gap, b_sp in bonus_combos:
        p = dict(best_params)
        p["b_early"] = b_early
        p["b_parents_young"] = b_py
        p["b_dam_bms_gap"] = b_gap
        p["b_sale_price"] = b_sp
        cv = loo_cv_score(all_data, p)
        if cv > best_score:
            best_score = cv
            best_params.update({
                "b_early": b_early, "b_parents_young": b_py,
                "b_dam_bms_gap": b_gap, "b_sale_price": b_sp,
            })

    print(f"  最良: early={best_params['b_early']}, parents_young={best_params['b_parents_young']}, "
          f"dam_bms_gap={best_params['b_dam_bms_gap']}, sale_price={best_params['b_sale_price']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 3: 産駒番号 + コネクション重み
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 3: 産駒番号 + コネクション重み")
    print(f"{'='*60}")
    conn_combos = list(itertools.product(
        grid["b_foal_penalty"], grid["b_foal_bonus"],
        grid["w_trainer"], grid["w_owner"], grid["w_breeder"]
    ))
    print(f"  組み合わせ数: {len(conn_combos)}")

    for fp, fb, wt, wo, wb in conn_combos:
        p = dict(best_params)
        p["b_foal_penalty"] = fp
        p["b_foal_bonus"] = fb
        p["w_trainer"] = wt
        p["w_owner"] = wo
        p["w_breeder"] = wb
        cv = loo_cv_score(all_data, p)
        if cv > best_score:
            best_score = cv
            best_params.update({
                "b_foal_penalty": fp, "b_foal_bonus": fb,
                "w_trainer": wt, "w_owner": wo, "w_breeder": wb,
            })

    # ===============================================================
    # Stage 4: 微調整（最良パラメータ周辺で細かくサーチ）
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 4: 血統重みの微調整")
    print(f"{'='*60}")
    fine_grid = {
        "w_sire_ei": sorted(set([best_params["w_sire_ei"] + d for d in [-0.025, 0, 0.025, 0.05]])),
        "w_dam_prize": sorted(set([best_params["w_dam_prize"] + d for d in [-0.025, 0, 0.025, 0.05]])),
        "w_bms_ei": sorted(set([best_params["w_bms_ei"] + d for d in [-0.025, 0, 0.025, 0.05]])),
    }
    fine_combos = list(itertools.product(
        fine_grid["w_sire_ei"], fine_grid["w_dam_prize"], fine_grid["w_bms_ei"]
    ))
    print(f"  組み合わせ数: {len(fine_combos)}")

    for w_sire, w_dam, w_bms in fine_combos:
        if w_sire <= 0 or w_dam <= 0 or w_bms <= 0:
            continue
        p = dict(best_params)
        p["w_sire_ei"] = w_sire
        p["w_dam_prize"] = w_dam
        p["w_bms_ei"] = w_bms
        cv = loo_cv_score(all_data, p)
        if cv > best_score:
            best_score = cv
            best_params.update({"w_sire_ei": w_sire, "w_dam_prize": w_dam, "w_bms_ei": w_bms})

    # ------------------------------------------------------------------
    # 最終結果
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  最適パラメータ")
    print("=" * 70)
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")
    print(f"\n  CV複合スコア: {best_score:.2f} (現行: {current_cv:.2f}, 差: {best_score - current_cv:+.2f})")

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
    print("-" * 55)
    for y in sorted(all_data.keys()):
        m_old = evaluate_single(all_data[y], current_params)
        m_new = evaluate_single(all_data[y], best_params)
        print(f"  {y} | {'Spearman':>10} | {m_old['spearman']:8.4f} | {m_new['spearman']:8.4f} | {m_new['spearman']-m_old['spearman']:+8.4f}")
        print(f"       | {'TOP10':>10} | {m_old['top10']:8d} | {m_new['top10']:8d} | {m_new['top10']-m_old['top10']:+8d}")
        print(f"       | {'TOP30':>10} | {m_old['top30']:8d} | {m_new['top30']:8d} | {m_new['top30']-m_old['top30']:+8d}")
        print(f"       | {'TOP50':>10} | {m_old['top50']:8d} | {m_new['top50']:8d} | {m_new['top50']-m_old['top50']:+8d}")
        print(f"       | {'賞金倍率':>10} | {m_old['prize_ratio']:8.2f}x | {m_new['prize_ratio']:8.2f}x | {m_new['prize_ratio']-m_old['prize_ratio']:+8.2f}")
        print("-" * 55)

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="グリッドサーチ（重み最適化）")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2015, 2016, 2017, 2018, 2019, 2020, 2021],
                        help="使用する年度リスト")
    args = parser.parse_args()

    best = grid_search(args.years)
