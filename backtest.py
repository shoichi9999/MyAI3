"""
バックテスト — ダービー/オークスTOP5予測の精度検証。

指定年のデータで特徴量を構築し、ヒューリスティックスコアによる
予測ランキングがクラシックTOP5をどれだけ捉えられたかを評価する。

使い方:
  python backtest.py 2021
  python backtest.py --all
  python backtest.py --race oaks --all      # オークス評価
  python backtest.py --race derby --all     # ダービー評価（デフォルト）
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

_CLASSIC_CACHE: dict | None = None


def _load_classic_results() -> dict:
    """data/classic_results.json を読み込む（キャッシュ付き）。"""
    global _CLASSIC_CACHE
    if _CLASSIC_CACHE is not None:
        return _CLASSIC_CACHE
    path = "data/classic_results.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません")
    with open(path, "r", encoding="utf-8") as f:
        _CLASSIC_CACHE = json.load(f)
    return _CLASSIC_CACHE


def get_classic_top5(birth_year: int, race_type: str) -> list[str]:
    """指定生年・レース種別のTOP5 horse_id リストを返す。"""
    data = _load_classic_results()
    return data.get(race_type, {}).get(str(birth_year), [])


# ------------------------------------------------------------------
# データ準備
# ------------------------------------------------------------------

def load_backtest_data(year: int, race_type: str = "derby") -> pd.DataFrame:
    """バックテスト用データを読み込み、特徴量を生成する。"""
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません")

    horses = pd.read_csv(csv_path)
    sex_filter = "牡" if race_type == "derby" else "牝"
    features = build_feature_matrix(horses, birth_year=year, sex_filter=sex_filter)

    return features


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_classic(df: pd.DataFrame, scores: np.ndarray,
                     birth_year: int, race_type: str = "derby") -> dict:
    """クラシックTOP5に対する予測精度を評価する。"""
    horse_ids = df["horse_id"].astype(str).values

    classic_top5 = set(get_classic_top5(birth_year, race_type))

    results = {"birth_year": birth_year}

    # 予測TOP N にクラシックTOP5が何頭入るか
    for n in [5, 10, 15, 20, 30]:
        pred_top_idx = np.argsort(-scores)[:n]
        pred_top_ids = set(horse_ids[pred_top_idx])
        results[f"top{n}_{race_type}"] = len(pred_top_ids & classic_top5)

    # クラシック馬の平均予測順位
    all_ranks = np.argsort(np.argsort(-scores)) + 1  # 1-indexed
    classic_ranks = []
    for hid in classic_top5:
        idx = np.where(horse_ids == hid)[0]
        if len(idx) > 0:
            classic_ranks.append(int(all_ranks[idx[0]]))
    results[f"{race_type}_avg_rank"] = np.mean(classic_ranks) if classic_ranks else float("nan")
    results[f"{race_type}_median_rank"] = np.median(classic_ranks) if classic_ranks else float("nan")
    results[f"{race_type}_ranks"] = sorted(classic_ranks)

    return results


def print_metrics(m: dict, race_type: str = "derby"):
    year = m["birth_year"]
    race_year = year + 3
    race_label = "ダービー" if race_type == "derby" else "オークス"
    sex_label = "牡馬" if race_type == "derby" else "牝馬"
    print(f"\n  --- {year}年産 ({race_label} {race_year}年) ---")

    print(f"  【{sex_label}】予測TOP N に{race_label}TOP5が何頭入るか:")
    for n in [5, 10, 15, 20, 30]:
        d = m.get(f"top{n}_{race_type}", 0)
        print(f"    TOP{n:>3d}: {d}/5")

    avg = m.get(f"{race_type}_avg_rank", float("nan"))
    med = m.get(f"{race_type}_median_rank", float("nan"))
    print(f"  {race_label}馬の予測順位: 平均={avg:.0f}位  中央値={med:.0f}位")
    ranks = m.get(f"{race_type}_ranks", [])
    if ranks:
        print(f"    個別: {ranks}")


def print_top_horses(df: pd.DataFrame, birth_year: int, race_type: str = "derby",
                     n: int = 30):
    """予測TOP N を表示し、クラシック馬をハイライトする。"""
    classic_top5 = set(get_classic_top5(birth_year, race_type))
    race_label = "ダービー" if race_type == "derby" else "オークス"
    sex_label = "牡馬" if race_type == "derby" else "牝馬"

    df = df.copy()
    df["pred_rank"] = df["score"].rank(ascending=False).astype(int)

    print(f"\n--- 予測 TOP{n}（{sex_label}） ---")
    top_pred = df.nlargest(n, "score")
    rows = []
    for i, (_, r) in enumerate(top_pred.iterrows(), 1):
        hid = str(r["horse_id"])
        mark = ""
        if hid in classic_top5:
            pos = list(get_classic_top5(birth_year, race_type)).index(hid) + 1
            mark = f"★{race_label}{pos}着"
        rows.append([
            i, r["horse_name"], f'{r["score"]:.1f}', mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "スコア", f"{race_label}成績"],
                   tablefmt="simple"))

    # クラシック馬の予測順位一覧
    race_year = birth_year + 3
    print(f"\n--- {race_year}年{race_label}TOP5の予測順位 ---")
    for i, hid in enumerate(get_classic_top5(birth_year, race_type), 1):
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
    parser.add_argument("--race", choices=["derby", "oaks"],
                        default="derby",
                        help="対象レース: derby(ダービー・牡馬) / oaks(オークス・牝馬)")
    args = parser.parse_args()

    if args.all:
        years = list(range(2015, 2023))
    elif args.year:
        years = [args.year]
    else:
        parser.error("年度を指定するか --all を使用してください")

    race_type = args.race
    race_label = "ダービー" if race_type == "derby" else "オークス"
    sex_label = "牡馬" if race_type == "derby" else "牝馬"

    print("=" * 60)
    print(f"  バックテスト: {sex_label}{race_label}TOP5予測")
    print("=" * 60)

    all_metrics = []
    for year in years:
        classic = get_classic_top5(year, race_type)
        if not classic:
            print(f"\n  {year}年: クラシック結果データなし — スキップ")
            continue

        df = load_backtest_data(year, race_type=race_type)
        scores = heuristic_score(df, race_type=race_type).values
        df["score"] = scores

        metrics = evaluate_classic(df, scores, year, race_type=race_type)
        all_metrics.append(metrics)
        print_metrics(metrics, race_type=race_type)
        print_top_horses(df, year, race_type=race_type)

    # 全年度サマリ
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print(f"  全年度サマリ（{sex_label}{race_label}TOP5）")
        print("=" * 60)

        headers = ["生年", "TOP5", "TOP10", "TOP15", "TOP20", "TOP30", "平均順位"]
        rows = []
        for m in all_metrics:
            rows.append([
                m["birth_year"],
                f'{m.get(f"top5_{race_type}", 0)}/5',
                f'{m.get(f"top10_{race_type}", 0)}/5',
                f'{m.get(f"top15_{race_type}", 0)}/5',
                f'{m.get(f"top20_{race_type}", 0)}/5',
                f'{m.get(f"top30_{race_type}", 0)}/5',
                f'{m.get(f"{race_type}_avg_rank", 0):.0f}',
            ])

        # 平均行
        avg_row = ["平均"]
        for key in [f"top5_{race_type}", f"top10_{race_type}", f"top15_{race_type}",
                    f"top20_{race_type}", f"top30_{race_type}"]:
            avg = np.mean([m.get(key, 0) for m in all_metrics])
            avg_row.append(f"{avg:.1f}/5")
        avg_row.append(f'{np.mean([m.get(f"{race_type}_avg_rank", 0) for m in all_metrics]):.0f}')
        rows.append(avg_row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
