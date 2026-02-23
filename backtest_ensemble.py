"""
アンサンブルバックテスト — ヒューリスティック × LightGBM。

Leave-One-Year-Out CV で LightGBM を学習し、
ヒューリスティックとのアンサンブルで精度を評価する。

使い方:
  python backtest_ensemble.py 2021
  python backtest_ensemble.py 2021 --alpha 0.3
  python backtest_ensemble.py --all
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabulate import tabulate

from src.features import build_feature_matrix
from src.model import heuristic_score
from src.ml_model import FEATURE_COLS, train_predict_loyo, ensemble_score


YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


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


def evaluate(y_true, scores):
    corr, pval = spearmanr(scores, y_true)
    results = {}
    for n in [10, 30, 50, 100]:
        pred_idx = set(np.argsort(-scores)[:n])
        actual_idx = set(np.argsort(-y_true)[:n])
        results[f"top{n}"] = len(pred_idx & actual_idx)
    overall_avg = y_true.mean()
    pred_top30_avg = y_true[np.argsort(-scores)[:30]].mean()
    return {
        "spearman": corr,
        "pval": pval,
        "top10": results["top10"],
        "top30": results["top30"],
        "top50": results["top50"],
        "top100": results["top100"],
        "prize_ratio": pred_top30_avg / overall_avg if overall_avg > 0 else 0,
    }


def print_comparison(year, m_h, m_ml, m_ens):
    """3手法の比較表示。"""
    print(f"\n{'='*70}")
    print(f"  {year}年産駒 — 手法比較")
    print(f"{'='*70}")
    header = f"{'指標':>12} | {'Heuristic':>10} | {'LightGBM':>10} | {'Ensemble':>10}"
    print(header)
    print("-" * 60)
    for key, label in [("spearman", "Spearman"), ("top10", "TOP10"),
                       ("top30", "TOP30"), ("top50", "TOP50"),
                       ("top100", "TOP100"), ("prize_ratio", "賞金倍率")]:
        h_val = m_h[key]
        ml_val = m_ml[key]
        e_val = m_ens[key]
        if key == "spearman":
            print(f"  {label:>10} | {h_val:10.4f} | {ml_val:10.4f} | {e_val:10.4f}")
        elif key == "prize_ratio":
            print(f"  {label:>10} | {h_val:9.2f}x | {ml_val:9.2f}x | {e_val:9.2f}x")
        else:
            print(f"  {label:>10} | {h_val:10d} | {ml_val:10d} | {e_val:10d}")
    print("-" * 60)


def print_top_horses(df, score_col, label, n=20):
    """予測 TOP N を表示。"""
    df = df.copy()
    df["pred_rank"] = df[score_col].rank(ascending=False).astype(int)
    df["actual_rank"] = df["prize_num"].rank(ascending=False).astype(int)

    print(f"\n--- {label} 予測 TOP{n} ---")
    top_pred = df.nlargest(n, score_col)
    rows = []
    for i, (_, r) in enumerate(top_pred.iterrows(), 1):
        mark = "*" if r["actual_rank"] <= 30 else ""
        rows.append([
            i, r["horse_name"], f'{r[score_col]:.3f}',
            f'{r["prize_num"]:,.0f}万', r["actual_rank"], mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "スコア", "実賞金", "実順位", ""],
                   tablefmt="simple"))


def run_backtest(target_year, all_data, alpha=0.5):
    """1年分のバックテスト。"""
    df = all_data[target_year]
    y_true = df["prize_num"].values

    # (1) Heuristic
    h_scores = heuristic_score(df).values

    # (2) LightGBM (LOYO) — TOP50分類 + スタッキング
    ml_proba = train_predict_loyo(all_data, target_year, top_n=50)

    # (3) Ensemble
    ens_scores = ensemble_score(h_scores, ml_proba, alpha=alpha)

    m_h = evaluate(y_true, h_scores)
    m_ml = evaluate(y_true, ml_proba)
    m_ens = evaluate(y_true, ens_scores)

    print_comparison(target_year, m_h, m_ml, m_ens)

    # アンサンブル TOP20 を表示
    df = df.copy()
    df["h_score"] = h_scores
    df["ml_proba"] = ml_proba
    df["ens_score"] = ens_scores
    print_top_horses(df, "ens_score", "Ensemble", n=20)

    return m_h, m_ml, m_ens


def main():
    parser = argparse.ArgumentParser(description="アンサンブルバックテスト")
    parser.add_argument("year", type=int, nargs="?", default=None,
                        help="評価対象年（省略時は --all が必要）")
    parser.add_argument("--all", action="store_true",
                        help="全年度で実行して集計")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="ヒューリスティックの重み (0=ML only, 1=heuristic only)")
    args = parser.parse_args()

    if args.year is None and not args.all:
        parser.error("year を指定するか --all を使用してください")

    # データ読み込み
    print("=== データ読み込み ===")
    all_data = {}
    for y in YEARS:
        df = load_year(y)
        if df is not None:
            all_data[y] = df
            print(f"  {y}年: {len(df)}頭")

    # スタッキング: ヒューリスティックスコアを全年度に事前追加
    for y, df in all_data.items():
        all_data[y] = df.copy()
        all_data[y]["h_score"] = heuristic_score(df).values

    print(f"\n  alpha = {args.alpha} (H={args.alpha:.0%}, ML={1-args.alpha:.0%})")

    if args.all:
        # 全年度バックテスト
        sum_h = {"top10": 0, "top30": 0, "top50": 0, "top100": 0,
                 "spearman": 0, "prize_ratio": 0}
        sum_ml = dict(sum_h)
        sum_ens = dict(sum_h)
        n_years = 0

        for y in sorted(all_data.keys()):
            m_h, m_ml, m_ens = run_backtest(y, all_data, alpha=args.alpha)
            for key in sum_h:
                sum_h[key] += m_h[key]
                sum_ml[key] += m_ml[key]
                sum_ens[key] += m_ens[key]
            n_years += 1

        # 集計
        print(f"\n{'='*70}")
        print(f"  全{n_years}年度 集計")
        print(f"{'='*70}")
        header = f"{'指標':>12} | {'Heuristic':>10} | {'LightGBM':>10} | {'Ensemble':>10}"
        print(header)
        print("-" * 60)
        for key, label in [("spearman", "Spearman平均"), ("top10", "TOP10合計"),
                           ("top30", "TOP30合計"), ("top50", "TOP50合計"),
                           ("top100", "TOP100合計"), ("prize_ratio", "賞金倍率平均")]:
            if key in ("spearman", "prize_ratio"):
                print(f"  {label:>10} | {sum_h[key]/n_years:10.4f} | "
                      f"{sum_ml[key]/n_years:10.4f} | {sum_ens[key]/n_years:10.4f}")
            else:
                print(f"  {label:>10} | {sum_h[key]:10d} | "
                      f"{sum_ml[key]:10d} | {sum_ens[key]:10d}")
        print("-" * 60)

    else:
        if args.year not in all_data:
            print(f"[ERROR] {args.year}年のデータがありません")
            return
        run_backtest(args.year, all_data, alpha=args.alpha)


if __name__ == "__main__":
    main()
