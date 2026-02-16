"""
重みの最適化スクリプト。

2022年世代のバックテストデータを使い、
Spearman順位相関が最大化される特徴量の重み配分を探索する。

手法:
1. グリッドサーチ（粗い→細かい2段階）
2. scipy.optimize による数値最適化
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from itertools import product

from src.features import (
    get_sire_ei,
    get_bms_ei,
    get_dam_prize,
    calc_trainer_score,
    calc_breeder_score,
    DAM_PRIZES,
)


def load_backtest_data(year: int = 2022) -> pd.DataFrame:
    """バックテスト用データを読み込み、特徴量を計算する。"""
    horses = pd.read_csv(f"data/horses_{year}.csv")

    # 実績賞金（目的変数）
    horses["prize_num"] = horses["total_prize"].apply(
        lambda x: float(str(x).replace(",", "").replace("万", "").strip())
        if pd.notna(x) and str(x).strip()
        else 0.0
    )

    # 特徴量の生値を計算
    horses["sire_ei"] = horses["sire"].apply(
        lambda x: get_sire_ei(str(x)) if pd.notna(x) else 0
    )
    horses["bms_ei"] = horses["sire_of_dam"].apply(
        lambda x: get_bms_ei(str(x)) if pd.notna(x) else 0
    )

    # 母馬賞金（中央値補完）
    dam_prizes_list = [v for v in DAM_PRIZES.values() if v > 0]
    dam_median = np.median(dam_prizes_list) if dam_prizes_list else 0.0
    horses["dam_prize"] = horses["dam"].apply(
        lambda x: get_dam_prize(str(x)) if pd.notna(x) else 0
    )
    horses["dam_prize"] = horses["dam_prize"].apply(
        lambda x: x if x > 0 else dam_median
    )

    horses["trainer_score"] = horses["trainer"].apply(
        lambda x: calc_trainer_score(str(x)) if pd.notna(x) else 50
    )
    horses["breeder_score"] = horses["breeder"].apply(
        lambda x: calc_breeder_score(str(x)) if pd.notna(x) else 50
    )

    return horses


def calc_score(horses: pd.DataFrame, weights: dict) -> pd.Series:
    """重みに基づいてスコアを計算する。"""
    # 正規化
    sire_ei = horses["sire_ei"].fillna(0)
    bms_ei = horses["bms_ei"].fillna(0)
    dam_log = np.log1p(horses["dam_prize"].fillna(0))
    trainer = horses["trainer_score"].fillna(50)
    breeder = horses["breeder_score"].fillna(50)

    sire_max = sire_ei.max()
    bms_max = bms_ei.max()
    dam_max = dam_log.max()

    score = pd.Series(0.0, index=horses.index)

    if sire_max > 0:
        score += (sire_ei / sire_max * 100) * weights["sire_ei"]
    if bms_max > 0:
        score += (bms_ei / bms_max * 100) * weights["bms_ei"]
    if dam_max > 0:
        score += (dam_log / dam_max * 100) * weights["dam_prize"]
    score += trainer * weights["trainer"]
    score += breeder * weights["breeder"]

    return score


def evaluate(horses: pd.DataFrame, weights: dict) -> dict:
    """重みの評価指標を計算する。"""
    score = calc_score(horses, weights)
    corr, pval = spearmanr(score, horses["prize_num"])

    # TOP30一致率
    horses_eval = horses.copy()
    horses_eval["_score"] = score
    pred_top30_ids = set(horses_eval.nlargest(30, "_score")["horse_id"].values)
    actual_top30_ids = set(horses_eval.nlargest(30, "prize_num")["horse_id"].values)
    overlap_30 = len(pred_top30_ids & actual_top30_ids)

    # 予測TOP30の平均賞金 / 全体平均賞金
    pred_avg = horses_eval.loc[
        horses_eval["horse_id"].isin(pred_top30_ids), "prize_num"
    ].mean()
    overall_avg = horses_eval["prize_num"].mean()
    prize_ratio = pred_avg / overall_avg if overall_avg > 0 else 0

    return {
        "spearman": corr,
        "pval": pval,
        "top30_overlap": overlap_30,
        "prize_ratio": prize_ratio,
    }


def grid_search(horses: pd.DataFrame, step: float = 0.05) -> list:
    """グリッドサーチで最適な重みを探索する。"""
    results = []
    values = np.arange(0, 1.01, step)

    # 5変数の組み合わせ（合計=1.0）
    for w_sire in values:
        for w_dam in values:
            for w_bms in values:
                for w_trainer in values:
                    w_breeder = round(1.0 - w_sire - w_dam - w_bms - w_trainer, 2)
                    if w_breeder < -0.001 or w_breeder > 1.001:
                        continue
                    w_breeder = max(0, w_breeder)

                    weights = {
                        "sire_ei": w_sire,
                        "dam_prize": w_dam,
                        "bms_ei": w_bms,
                        "trainer": w_trainer,
                        "breeder": w_breeder,
                    }
                    metrics = evaluate(horses, weights)
                    metrics["weights"] = weights
                    results.append(metrics)

    results.sort(key=lambda x: x["spearman"], reverse=True)
    return results


def optimize_scipy(horses: pd.DataFrame) -> dict:
    """scipy.optimizeで最適な重みを探索する。"""

    def objective(params):
        # params: [sire_ei, dam_prize, bms_ei, trainer] (breeder = 残り)
        w = np.abs(params)  # 非負制約
        total = w.sum()
        if total == 0:
            return 0
        w = w / total  # 合計1に正規化

        weights = {
            "sire_ei": w[0],
            "dam_prize": w[1],
            "bms_ei": w[2],
            "trainer": w[3],
            "breeder": w[4],
        }
        metrics = evaluate(horses, weights)
        return -metrics["spearman"]  # 最小化なので符号反転

    # 複数の初期値で試す
    best_result = None
    best_corr = -1

    initial_points = [
        [0.20, 0.30, 0.10, 0.25, 0.05],  # 現在の重み
        [0.25, 0.25, 0.15, 0.30, 0.05],
        [0.15, 0.35, 0.10, 0.30, 0.10],
        [0.30, 0.20, 0.20, 0.25, 0.05],
        [0.10, 0.40, 0.05, 0.40, 0.05],
        [0.20, 0.20, 0.20, 0.20, 0.20],
        [0.05, 0.50, 0.05, 0.35, 0.05],
        [0.40, 0.10, 0.10, 0.30, 0.10],
    ]

    for x0 in initial_points:
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 0.001, "fatol": 1e-6},
        )
        if -result.fun > best_corr:
            best_corr = -result.fun
            best_result = result

    # 結果の重みを正規化
    w = np.abs(best_result.x)
    w = w / w.sum()

    return {
        "sire_ei": round(w[0], 3),
        "dam_prize": round(w[1], 3),
        "bms_ei": round(w[2], 3),
        "trainer": round(w[3], 3),
        "breeder": round(w[4], 3),
    }


def main():
    print("=" * 70)
    print("  重みの最適化（2022年世代バックテスト）")
    print("=" * 70)

    horses = load_backtest_data(2022)
    print(f"データ: {len(horses)}頭")

    # === 現在の重み ===
    current = {
        "sire_ei": 0.20,
        "dam_prize": 0.30,
        "bms_ei": 0.10,
        "trainer": 0.25,
        "breeder": 0.05,
    }
    current_metrics = evaluate(horses, current)
    print(f"\n--- 現在の重み ---")
    print(f"  重み: {current}")
    print(f"  Spearman: {current_metrics['spearman']:.4f}")
    print(f"  TOP30一致: {current_metrics['top30_overlap']}/30")
    print(f"  賞金倍率: {current_metrics['prize_ratio']:.2f}x")

    # === グリッドサーチ（粗い: 10%刻み） ===
    print(f"\n--- グリッドサーチ（10%刻み） ---")
    results_coarse = grid_search(horses, step=0.10)
    print(f"  探索パターン: {len(results_coarse)}件")

    top5 = results_coarse[:5]
    print(f"\n  TOP5:")
    for i, r in enumerate(top5, 1):
        w = r["weights"]
        print(
            f"  {i}. Spearman={r['spearman']:.4f}  "
            f"TOP30={r['top30_overlap']}/30  "
            f"倍率={r['prize_ratio']:.2f}x  "
            f"重み: 父EI={w['sire_ei']:.2f} 母賞金={w['dam_prize']:.2f} "
            f"母父EI={w['bms_ei']:.2f} 調教師={w['trainer']:.2f} 牧場={w['breeder']:.2f}"
        )

    # === グリッドサーチ（細かい: 5%刻み、TOP1周辺） ===
    best_coarse = top5[0]["weights"]
    print(f"\n--- グリッドサーチ（5%刻み） ---")
    results_fine = grid_search(horses, step=0.05)
    print(f"  探索パターン: {len(results_fine)}件")

    top10 = results_fine[:10]
    print(f"\n  TOP10:")
    for i, r in enumerate(top10, 1):
        w = r["weights"]
        print(
            f"  {i:>2d}. Spearman={r['spearman']:.4f}  "
            f"TOP30={r['top30_overlap']}/30  "
            f"倍率={r['prize_ratio']:.2f}x  "
            f"重み: 父EI={w['sire_ei']:.2f} 母賞金={w['dam_prize']:.2f} "
            f"母父EI={w['bms_ei']:.2f} 調教師={w['trainer']:.2f} 牧場={w['breeder']:.2f}"
        )

    # === scipy最適化 ===
    print(f"\n--- 数値最適化（Nelder-Mead） ---")
    optimal = optimize_scipy(horses)
    opt_metrics = evaluate(horses, optimal)
    print(f"  最適重み: {optimal}")
    print(f"  Spearman: {opt_metrics['spearman']:.4f}")
    print(f"  TOP30一致: {opt_metrics['top30_overlap']}/30")
    print(f"  賞金倍率: {opt_metrics['prize_ratio']:.2f}x")

    # === 比較サマリー ===
    print(f"\n{'=' * 70}")
    print(f"  最適化結果サマリー")
    print(f"{'=' * 70}")

    best_grid = results_fine[0]
    print(f"\n  {'方法':<20s} {'Spearman':>10s} {'TOP30':>8s} {'倍率':>8s}")
    print(f"  {'-'*48}")
    print(
        f"  {'現在の重み':<20s} {current_metrics['spearman']:>10.4f} "
        f"{current_metrics['top30_overlap']:>5d}/30 "
        f"{current_metrics['prize_ratio']:>7.2f}x"
    )
    print(
        f"  {'グリッドサーチ最良':<20s} {best_grid['spearman']:>10.4f} "
        f"{best_grid['top30_overlap']:>5d}/30 "
        f"{best_grid['prize_ratio']:>7.2f}x"
    )
    print(
        f"  {'数値最適化最良':<20s} {opt_metrics['spearman']:>10.4f} "
        f"{opt_metrics['top30_overlap']:>5d}/30 "
        f"{opt_metrics['prize_ratio']:>7.2f}x"
    )

    improvement = (best_grid["spearman"] - current_metrics["spearman"]) / abs(
        current_metrics["spearman"]
    ) * 100
    print(f"\n  グリッドサーチによる改善: {improvement:+.1f}%")

    # 最適な重みの推奨
    best_w = best_grid["weights"]
    if opt_metrics["spearman"] > best_grid["spearman"]:
        best_w = optimal
        print(f"\n  推奨重み（数値最適化）:")
    else:
        print(f"\n  推奨重み（グリッドサーチ）:")

    print(f"    WEIGHT_SIRE_EI  = {best_w['sire_ei']:.2f}")
    print(f"    WEIGHT_DAM_PRIZE = {best_w['dam_prize']:.2f}")
    print(f"    WEIGHT_BMS_EI   = {best_w['bms_ei']:.2f}")
    print(f"    WEIGHT_TRAINER  = {best_w['trainer']:.2f}")
    print(f"    WEIGHT_BREEDER  = {best_w['breeder']:.2f}")


if __name__ == "__main__":
    main()
