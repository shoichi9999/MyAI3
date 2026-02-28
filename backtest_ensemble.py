"""
アンサンブルバックテスト — ヒューリスティック × LightGBM。

Leave-One-Year-Out CV で LightGBM を学習し、
ヒューリスティックとのアンサンブルで精度を評価する。
評価基準: 牡馬ダービーTOP5 のヒット数。

使い方:
  python backtest_ensemble.py 2021
  python backtest_ensemble.py 2021 --alpha 0.3
  python backtest_ensemble.py --all
  python backtest_ensemble.py --all --sweep   # alpha最適値を探索
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.features import build_feature_matrix
from src.model import heuristic_score
from src.ml_model import train_predict_loyo, ensemble_score


YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def _load_classic_results() -> dict:
    path = "data/classic_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_CLASSIC = _load_classic_results()


def _get_derby_ids(birth_year: int) -> set:
    return set(_CLASSIC.get("derby", {}).get(str(birth_year), []))


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
    # build_feature_matrix は牡馬のみにフィルタするため、horse_id で結合
    prize_map = dict(zip(horses["horse_id"].astype(str), horses["prize_num"]))
    features["prize_num"] = features["horse_id"].astype(str).map(prize_map).fillna(0.0)
    return features


def evaluate_classic(scores, horse_ids, birth_year):
    """牡馬ダービーTOP5ヒットを評価する。"""
    derby_top5 = _get_derby_ids(birth_year)

    results = {}
    for n in [5, 10, 15, 20, 30]:
        pred_idx = np.argsort(-scores)[:n]
        pred_ids = set(horse_ids[pred_idx])
        results[f"top{n}_derby"] = len(pred_ids & derby_top5)

    return results


def run_backtest(target_year, all_data, alpha=0.5, verbose=True):
    """1年分のバックテスト（牡馬ダービー結果で評価）。"""
    df = all_data[target_year]
    horse_ids = df["horse_id"].astype(str).values

    # (1) Heuristic
    h_scores = heuristic_score(df).values

    # (2) LightGBM (LOYO) — TOP50分類 + スタッキング
    ml_proba = train_predict_loyo(all_data, target_year, top_n=50)

    # (3) Ensemble
    ens_scores = ensemble_score(h_scores, ml_proba, alpha=alpha)

    m_h = evaluate_classic(h_scores, horse_ids, target_year)
    m_ml = evaluate_classic(ml_proba, horse_ids, target_year)
    m_ens = evaluate_classic(ens_scores, horse_ids, target_year)

    if verbose:
        derby_top5 = _get_derby_ids(target_year)
        print(f"\n{'='*75}")
        print(f"  {target_year}年産駒（牡馬） — ダービー予測比較 (alpha={alpha})")
        print(f"{'='*75}")
        header = f"{'指標':>16} | {'Heuristic':>10} | {'LightGBM':>10} | {'Ensemble':>10}"
        print(header)
        print("-" * 65)
        for key, label in [
            ("top5_derby", "TOP5 Derby"),
            ("top10_derby", "TOP10 Derby"),
            ("top15_derby", "TOP15 Derby"),
            ("top20_derby", "TOP20 Derby"),
            ("top30_derby", "TOP30 Derby"),
        ]:
            h_val = m_h.get(key, 0)
            ml_val = m_ml.get(key, 0)
            e_val = m_ens.get(key, 0)
            best_val = max(h_val, ml_val, e_val)
            h_mark = " *" if h_val == best_val and h_val > 0 else ""
            ml_mark = " *" if ml_val == best_val and ml_val > 0 else ""
            e_mark = " *" if e_val == best_val and e_val > 0 else ""
            print(f"  {label:>14} | {h_val:>8d}{h_mark:2s} | "
                  f"{ml_val:>8d}{ml_mark:2s} | {e_val:>8d}{e_mark:2s}")
        print("-" * 65)

        # アンサンブル TOP20 とダービー馬の位置
        ens_rank = np.argsort(np.argsort(-ens_scores)) + 1
        print(f"\n  ダービー馬のアンサンブル予測順位:")
        for hid in sorted(derby_top5):
            idx = np.where(horse_ids == hid)[0]
            if len(idx) > 0:
                name = df.iloc[idx[0]]["horse_name"]
                rank = ens_rank[idx[0]]
                mark = "<<" if rank <= 5 else "<" if rank <= 15 else ""
                print(f"    {name:　<12s} → {rank:>5d}位 {mark}")

    return m_h, m_ml, m_ens


def main():
    parser = argparse.ArgumentParser(description="アンサンブルバックテスト（牡馬ダービー評価）")
    parser.add_argument("year", type=int, nargs="?", default=None,
                        help="評価対象年（省略時は --all が必要）")
    parser.add_argument("--all", action="store_true",
                        help="全年度で実行して集計")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="ヒューリスティックの重み (0=ML only, 1=heuristic only)")
    parser.add_argument("--sweep", action="store_true",
                        help="alpha を 0.0〜1.0 で探索して最適値を表示")
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

    if args.sweep and args.all:
        # alpha探索モード
        print("\n=== Alpha Sweep（牡馬ダービーTOP5） ===")
        print(f"{'alpha':>6} | {'TOP5':>6} | {'TOP10':>6} | {'TOP20':>6} | {'TOP30':>6}")
        print("-" * 50)

        best_alpha = 0.5
        best_total = 0

        for alpha_val in [i * 0.1 for i in range(11)]:
            sum_d5 = sum_d10 = sum_d20 = sum_d30 = 0
            for y in sorted(all_data.keys()):
                _, _, m_ens = run_backtest(y, all_data, alpha=alpha_val, verbose=False)
                sum_d5 += m_ens.get("top5_derby", 0)
                sum_d10 += m_ens.get("top10_derby", 0)
                sum_d20 += m_ens.get("top20_derby", 0)
                sum_d30 += m_ens.get("top30_derby", 0)
            mark = " ←" if sum_d5 > best_total else ""
            if sum_d5 > best_total:
                best_total = sum_d5
                best_alpha = alpha_val
            print(f"  {alpha_val:.1f}  | {sum_d5:>3}/40 | {sum_d10:>3}/40 | "
                  f"{sum_d20:>3}/40 | {sum_d30:>3}/40{mark}")

        print(f"\n  最適 alpha = {best_alpha:.1f} (TOP5 = {best_total}/40)")

        # 最適alphaで詳細表示
        print(f"\n\n{'='*75}")
        print(f"  最適 alpha={best_alpha:.1f} での詳細結果")
        print(f"{'='*75}")
        for y in sorted(all_data.keys()):
            run_backtest(y, all_data, alpha=best_alpha, verbose=True)

    elif args.all:
        # 全年度バックテスト
        sum_keys = ["top5_derby", "top10_derby", "top15_derby",
                    "top20_derby", "top30_derby"]
        sum_h = {k: 0 for k in sum_keys}
        sum_ml = {k: 0 for k in sum_keys}
        sum_ens = {k: 0 for k in sum_keys}

        for y in sorted(all_data.keys()):
            m_h, m_ml, m_ens = run_backtest(y, all_data, alpha=args.alpha)
            for k in sum_keys:
                sum_h[k] += m_h.get(k, 0)
                sum_ml[k] += m_ml.get(k, 0)
                sum_ens[k] += m_ens.get(k, 0)

        print(f"\n{'='*75}")
        print(f"  全{len(all_data)}年度 集計（牡馬ダービーTOP5） (alpha={args.alpha})")
        print(f"{'='*75}")
        header = f"{'指標':>16} | {'Heuristic':>10} | {'LightGBM':>10} | {'Ensemble':>10}"
        print(header)
        print("-" * 65)
        for key, label in [
            ("top5_derby", "TOP5 Derby"),
            ("top10_derby", "TOP10 Derby"),
            ("top15_derby", "TOP15 Derby"),
            ("top20_derby", "TOP20 Derby"),
            ("top30_derby", "TOP30 Derby"),
        ]:
            h_val = sum_h[key]
            ml_val = sum_ml[key]
            e_val = sum_ens[key]
            best_val = max(h_val, ml_val, e_val)
            h_mark = " *" if h_val == best_val else ""
            ml_mark = " *" if ml_val == best_val else ""
            e_mark = " *" if e_val == best_val else ""
            print(f"  {label:>14} | {h_val:>8d}{h_mark:2s} | "
                  f"{ml_val:>8d}{ml_mark:2s} | {e_val:>8d}{e_mark:2s}")
        print("-" * 65)

    else:
        if args.year not in all_data:
            print(f"[ERROR] {args.year}年のデータがありません")
            return
        run_backtest(args.year, all_data, alpha=args.alpha)


if __name__ == "__main__":
    main()
