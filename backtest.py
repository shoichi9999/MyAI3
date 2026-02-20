"""
バックテスト — ヒューリスティックスコアの精度検証。

指定年の産駒データで特徴量を構築し、実際の賞金ランキングとの
相関・TOP N一致率を評価する。

使い方:
  python backtest.py 2021
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabulate import tabulate

from src.features import build_feature_matrix
from src.model import heuristic_score


# ------------------------------------------------------------------
# データ準備
# ------------------------------------------------------------------

def load_backtest_data(year: int) -> pd.DataFrame:
    """バックテスト用データを読み込み、特徴量を生成する。"""
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません")

    horses = pd.read_csv(csv_path)
    horses["prize_num"] = horses["total_prize"].apply(_parse_prize)

    features = build_feature_matrix(horses, birth_year=year)
    features["prize_num"] = horses["prize_num"].values

    return features


def _parse_prize(x) -> float:
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    return float(str(x).replace(",", "").replace("万", "").strip())


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """評価指標を計算する。"""
    corr, pval = spearmanr(scores, y_true)

    results = {}
    for n in [10, 30, 50]:
        pred_idx = np.argsort(-scores)[:n]
        actual_idx = np.argsort(-y_true)[:n]
        overlap = len(set(pred_idx) & set(actual_idx))
        results[f"top{n}"] = overlap

    overall_avg = y_true.mean()
    pred_top30_avg = y_true[np.argsort(-scores)[:30]].mean()

    return {
        "spearman": corr,
        "pval": pval,
        "top10": results["top10"],
        "top30": results["top30"],
        "top50": results["top50"],
        "pred_top30_avg": pred_top30_avg,
        "overall_avg": overall_avg,
        "prize_ratio": pred_top30_avg / overall_avg if overall_avg > 0 else 0,
    }


def print_metrics(m: dict):
    print(f"  Spearman:    {m['spearman']:.4f} (p={m['pval']:.2e})")
    print(f"  TOP10一致:   {m['top10']}/10")
    print(f"  TOP30一致:   {m['top30']}/30")
    print(f"  TOP50一致:   {m['top50']}/50")
    print(f"  賞金倍率:    {m['prize_ratio']:.2f}x (予測TOP30平均/全体平均)")


def print_top_horses(df: pd.DataFrame, n: int = 20):
    """予測TOP N と実績TOP N を表示する。"""
    df = df.copy()
    df["pred_rank"] = df["score"].rank(ascending=False).astype(int)
    df["actual_rank"] = df["prize_num"].rank(ascending=False).astype(int)

    print(f"\n--- 予測 TOP{n} ---")
    top_pred = df.nlargest(n, "score")
    rows = []
    for i, (_, r) in enumerate(top_pred.iterrows(), 1):
        mark = "*" if r["actual_rank"] <= 30 else ""
        rows.append([
            i, r["horse_name"], f'{r["score"]:.1f}',
            f'{r["prize_num"]:,.0f}万', r["actual_rank"], mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "スコア", "実賞金", "実順位", ""],
                   tablefmt="simple"))

    print(f"\n--- 実績 TOP{n} ---")
    top_actual = df.nlargest(n, "prize_num")
    rows = []
    for i, (_, r) in enumerate(top_actual.iterrows(), 1):
        mark = "*" if r["pred_rank"] <= 30 else ""
        rows.append([
            i, r["horse_name"], f'{r["prize_num"]:,.0f}万',
            f'{r["score"]:.1f}', r["pred_rank"], mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "実賞金", "スコア", "予測順位", ""],
                   tablefmt="simple"))


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="バックテスト（ヒューリスティック）")
    parser.add_argument("year", type=int, help="評価対象の生年（例: 2021）")
    args = parser.parse_args()

    target = args.year

    print("=" * 60)
    print(f"  バックテスト: {target}年産駒（ヒューリスティック）")
    print("=" * 60)

    df = load_backtest_data(target)
    scores = heuristic_score(df).values
    y_true = df["prize_num"].values

    metrics = evaluate(y_true, scores)
    print_metrics(metrics)

    df["score"] = scores
    print_top_horses(df)


if __name__ == "__main__":
    main()
