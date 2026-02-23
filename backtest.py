"""
バックテスト — ダービー・オークスTOP5予測の精度検証。

指定年の産駒データで特徴量を構築し、ヒューリスティックスコアによる
予測ランキングがダービー/オークスTOP5をどれだけ捉えられたかを評価する。

使い方:
  python backtest.py 2021
  python backtest.py --all
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.features import build_feature_matrix
from src.model import heuristic_score


# ------------------------------------------------------------------
# クラシック結果データ
# ------------------------------------------------------------------

def _load_classic_results() -> dict:
    """data/classic_results.json を読み込む。"""
    path = "data/classic_results.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_classic_top5(birth_year: int, race_type: str) -> list[str]:
    """指定生年・レース種別のTOP5 horse_id リストを返す。"""
    data = _load_classic_results()
    return data.get(race_type, {}).get(str(birth_year), [])


# ------------------------------------------------------------------
# データ準備
# ------------------------------------------------------------------

def load_backtest_data(year: int) -> pd.DataFrame:
    """バックテスト用データを読み込み、特徴量を生成する。"""
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません")

    horses = pd.read_csv(csv_path)
    features = build_feature_matrix(horses, birth_year=year)

    return features


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_classic(df: pd.DataFrame, scores: np.ndarray,
                     birth_year: int) -> dict:
    """ダービー・オークスTOP5に対する予測精度を評価する。

    牡馬→ダービー、牝馬→オークスで分けて評価し、合算する。
    """
    horse_ids = df["horse_id"].astype(str).values
    sex = df["sex"].values  # 1=牡, 0=牝

    derby_top5 = set(get_classic_top5(birth_year, "derby"))
    oaks_top5 = set(get_classic_top5(birth_year, "oaks"))
    classic_top10 = derby_top5 | oaks_top5  # 合計10頭

    results = {"birth_year": birth_year}

    # 全体評価: 予測TOP N に何頭のクラシック馬が含まれるか
    for n in [10, 20, 30, 50, 100]:
        pred_top_idx = np.argsort(-scores)[:n]
        pred_top_ids = set(horse_ids[pred_top_idx])
        results[f"top{n}_derby"] = len(pred_top_ids & derby_top5)
        results[f"top{n}_oaks"] = len(pred_top_ids & oaks_top5)
        results[f"top{n}_total"] = len(pred_top_ids & classic_top10)

    # 牡馬のみでダービー評価
    male_mask = sex == 1
    if male_mask.any() and derby_top5:
        male_scores = scores[male_mask]
        male_ids = horse_ids[male_mask]
        for n in [10, 20, 30]:
            pred_idx = np.argsort(-male_scores)[:n]
            pred_ids = set(male_ids[pred_idx])
            results[f"male_top{n}_derby"] = len(pred_ids & derby_top5)

    # 牝馬のみでオークス評価
    female_mask = sex == 0
    if female_mask.any() and oaks_top5:
        female_scores = scores[female_mask]
        female_ids = horse_ids[female_mask]
        for n in [10, 20, 30]:
            pred_idx = np.argsort(-female_scores)[:n]
            pred_ids = set(female_ids[pred_idx])
            results[f"female_top{n}_oaks"] = len(pred_ids & oaks_top5)

    # クラシック馬の平均予測順位
    all_ranks = np.argsort(np.argsort(-scores)) + 1  # 1-indexed
    classic_ranks = []
    for hid in classic_top10:
        idx = np.where(horse_ids == hid)[0]
        if len(idx) > 0:
            classic_ranks.append(int(all_ranks[idx[0]]))
    results["classic_avg_rank"] = np.mean(classic_ranks) if classic_ranks else float("nan")
    results["classic_median_rank"] = np.median(classic_ranks) if classic_ranks else float("nan")
    results["classic_ranks"] = sorted(classic_ranks)

    return results


def print_metrics(m: dict):
    year = m["birth_year"]
    race_year = year + 3
    print(f"\n  --- {year}年産 (ダービー/オークス {race_year}年) ---")

    # 全体
    print(f"  【全体】予測TOP N にクラシックTOP10（ダービー5 + オークス5）が何頭入るか:")
    for n in [10, 20, 30, 50, 100]:
        d = m.get(f"top{n}_derby", 0)
        o = m.get(f"top{n}_oaks", 0)
        t = m.get(f"top{n}_total", 0)
        print(f"    TOP{n:>3d}: ダービー {d}/5  オークス {o}/5  合計 {t}/10")

    # 性別別
    print(f"  【牡馬内】予測TOP N にダービーTOP5が何頭:")
    for n in [10, 20, 30]:
        val = m.get(f"male_top{n}_derby", "?")
        print(f"    TOP{n}: {val}/5")

    print(f"  【牝馬内】予測TOP N にオークスTOP5が何頭:")
    for n in [10, 20, 30]:
        val = m.get(f"female_top{n}_oaks", "?")
        print(f"    TOP{n}: {val}/5")

    avg = m.get("classic_avg_rank", float("nan"))
    med = m.get("classic_median_rank", float("nan"))
    print(f"  クラシック馬の予測順位: 平均={avg:.0f}位  中央値={med:.0f}位")
    ranks = m.get("classic_ranks", [])
    if ranks:
        print(f"    個別: {ranks}")


def print_top_horses(df: pd.DataFrame, birth_year: int, n: int = 30):
    """予測TOP N を表示し、クラシック馬をハイライトする。"""
    derby_top5 = set(get_classic_top5(birth_year, "derby"))
    oaks_top5 = set(get_classic_top5(birth_year, "oaks"))

    df = df.copy()
    df["pred_rank"] = df["score"].rank(ascending=False).astype(int)

    print(f"\n--- 予測 TOP{n} ---")
    top_pred = df.nlargest(n, "score")
    rows = []
    for i, (_, r) in enumerate(top_pred.iterrows(), 1):
        hid = str(r["horse_id"])
        sex_str = "牡" if r["sex"] == 1 else "牝"
        mark = ""
        if hid in derby_top5:
            pos = list(get_classic_top5(birth_year, "derby")).index(hid) + 1
            mark = f"★ダービー{pos}着"
        elif hid in oaks_top5:
            pos = list(get_classic_top5(birth_year, "oaks")).index(hid) + 1
            mark = f"★オークス{pos}着"
        rows.append([
            i, r["horse_name"], sex_str, f'{r["score"]:.1f}', mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "性", "スコア", "クラシック成績"],
                   tablefmt="simple"))

    # クラシック馬の予測順位一覧
    race_year = birth_year + 3
    print(f"\n--- {race_year}年ダービーTOP5の予測順位 ---")
    for i, hid in enumerate(get_classic_top5(birth_year, "derby"), 1):
        row = df[df["horse_id"].astype(str) == hid]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {i}着 {r['horse_name']:　<15s} → 予測{r['pred_rank']:>5d}位 (スコア {r['score']:.1f})")

    print(f"\n--- {race_year}年オークスTOP5の予測順位 ---")
    for i, hid in enumerate(get_classic_top5(birth_year, "oaks"), 1):
        row = df[df["horse_id"].astype(str) == hid]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {i}着 {r['horse_name']:　<15s} → 予測{r['pred_rank']:>5d}位 (スコア {r['score']:.1f})")


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="バックテスト（ダービー/オークスTOP5予測）")
    parser.add_argument("year", nargs="?", type=int,
                        help="評価対象の生年（例: 2021）")
    parser.add_argument("--all", action="store_true",
                        help="全年度（2015-2022）を一括評価")
    args = parser.parse_args()

    if args.all:
        years = list(range(2015, 2023))
    elif args.year:
        years = [args.year]
    else:
        parser.error("年度を指定するか --all を使用してください")

    print("=" * 60)
    print("  バックテスト: ダービー/オークスTOP5予測")
    print("=" * 60)

    all_metrics = []
    for year in years:
        classic = get_classic_top5(year, "derby") + get_classic_top5(year, "oaks")
        if not classic:
            print(f"\n  {year}年: クラシック結果データなし — スキップ")
            continue

        df = load_backtest_data(year)
        scores = heuristic_score(df).values
        df["score"] = scores

        metrics = evaluate_classic(df, scores, year)
        all_metrics.append(metrics)
        print_metrics(metrics)
        print_top_horses(df, year)

    # 全年度サマリ
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("  全年度サマリ")
        print("=" * 60)

        headers = ["生年", "TOP10", "TOP20", "TOP30", "TOP50", "TOP100",
                   "牡TOP10", "牝TOP10", "平均順位"]
        rows = []
        for m in all_metrics:
            rows.append([
                m["birth_year"],
                f'{m.get("top10_total", 0)}/10',
                f'{m.get("top20_total", 0)}/10',
                f'{m.get("top30_total", 0)}/10',
                f'{m.get("top50_total", 0)}/10',
                f'{m.get("top100_total", 0)}/10',
                f'{m.get("male_top10_derby", 0)}/5',
                f'{m.get("female_top10_oaks", 0)}/5',
                f'{m.get("classic_avg_rank", 0):.0f}',
            ])

        # 平均行
        avg_row = ["平均"]
        for key in ["top10_total", "top20_total", "top30_total",
                    "top50_total", "top100_total"]:
            avg = np.mean([m.get(key, 0) for m in all_metrics])
            avg_row.append(f"{avg:.1f}/10")
        avg_row.append(f'{np.mean([m.get("male_top10_derby", 0) for m in all_metrics]):.1f}/5')
        avg_row.append(f'{np.mean([m.get("female_top10_oaks", 0) for m in all_metrics]):.1f}/5')
        avg_row.append(f'{np.mean([m.get("classic_avg_rank", 0) for m in all_metrics]):.0f}')
        rows.append(avg_row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
