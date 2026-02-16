"""
POG TOP10予測 - メイン実行モジュール。

使い方:
  1. データ収集モード: 過去〜当年のデータをスクレイピング
  2. 学習モード: 過去世代のデータでモデルを学習
  3. 予測モード: 当年の2歳馬からTOP10を予測
  4. 全自動モード: 上記すべてを一括実行
"""

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from src.scraper import scrape_all_2yo_data
from src.features import build_feature_matrix
from src.model import POGPredictor


def collect_data(birth_year: int, max_horses: int = None) -> dict:
    """データを収集してCSVに保存する。"""
    print(f"\n{'='*60}")
    print(f"  データ収集: {birth_year}年生まれの2歳馬")
    print(f"{'='*60}")
    return scrape_all_2yo_data(birth_year, max_horses=max_horses)


def train_model(training_years: list[int]) -> POGPredictor:
    """
    過去世代のデータでモデルを学習する。

    Parameters
    ----------
    training_years : list[int]
        学習に使う世代の生年リスト

    Returns
    -------
    POGPredictor
        学習済みモデル
    """
    print(f"\n{'='*60}")
    print(f"  モデル学習: {training_years}")
    print(f"{'='*60}")

    all_features = []

    for year in training_years:
        horses_path = f"data/horses_{year}.csv"

        if not os.path.exists(horses_path):
            print(f"[WARN] {horses_path} が見つかりません。スキップします。")
            continue

        horses_df = pd.read_csv(horses_path)
        features = build_feature_matrix(horses_df)
        features["birth_year"] = year
        all_features.append(features)

    if not all_features:
        print("[ERROR] 学習データがありません。先にデータを収集してください。")
        predictor = POGPredictor()
        return predictor

    training_df = pd.concat(all_features, ignore_index=True)
    print(f"学習データ: {len(training_df)}件")

    predictor = POGPredictor()
    predictor.train(training_df, target_col="total_earned")
    predictor.save()

    return predictor


def predict_top10(
    predictor: POGPredictor,
    target_year: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    当年の2歳馬からTOP N予測を行う。

    Parameters
    ----------
    predictor : POGPredictor
        予測モデル
    target_year : int
        対象の生年
    top_n : int
        上位何頭を出力するか

    Returns
    -------
    pd.DataFrame
        予測結果（上位N頭）
    """
    print(f"\n{'='*60}")
    print(f"  POG予測: {target_year}年生まれ TOP{top_n}")
    print(f"{'='*60}")

    horses_path = f"data/horses_{target_year}.csv"

    if not os.path.exists(horses_path):
        print(f"[ERROR] {horses_path} が見つかりません。先にデータを収集してください。")
        return pd.DataFrame()

    horses_df = pd.read_csv(horses_path)
    features = build_feature_matrix(horses_df)
    predictions = predictor.predict(features)

    # スコアでソートしてTOP N
    predictions = predictions.sort_values("ensemble_score", ascending=False)
    top = predictions.head(top_n).copy()

    # 表示用に整形
    display_cols = [
        "horse_name",
        "ensemble_score",
        "sire_ei",
        "bms_ei",
        "dam_prize",
    ]
    available_cols = [c for c in display_cols if c in top.columns]
    display_df = top[available_cols].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "順位"

    # カラム名を日本語に
    col_rename = {
        "horse_name": "馬名",
        "ensemble_score": "予測スコア",
        "sire_ei": "父EI",
        "bms_ei": "母父EI",
        "dam_prize": "母馬賞金(万)",
    }
    display_df = display_df.rename(columns=col_rename)

    print("\n" + tabulate(display_df, headers="keys", tablefmt="grid", floatfmt=".1f"))

    # 結果をCSVに保存
    output_path = f"data/pog_top{top_n}_{target_year}.csv"
    top.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果を保存しました: {output_path}")

    return top


def run_full_pipeline(
    target_year: int = 2024,
    training_years: list[int] = None,
    max_horses: int = None,
    top_n: int = 10,
):
    """
    全自動パイプライン。

    Parameters
    ----------
    target_year : int
        予測対象の世代（生年）
    training_years : list[int]
        学習に使う過去世代
    max_horses : int
        取得する最大馬数（テスト用）
    top_n : int
        上位何頭を出力するか
    """
    if training_years is None:
        training_years = list(range(target_year - 5, target_year))

    print("=" * 60)
    print("  POG予測システム - 全自動パイプライン")
    print(f"  対象世代: {target_year}年生まれ")
    print(f"  学習世代: {training_years}")
    print(f"  予測馬数: TOP{top_n}")
    print("=" * 60)

    # 1. データ収集（過去世代）
    for year in training_years:
        if not os.path.exists(f"data/horses_{year}.csv"):
            print(f"\n--- {year}年世代のデータを収集 ---")
            collect_data(year, max_horses=max_horses)
        else:
            print(f"\n--- {year}年世代: データ既存。スキップ ---")

    # 2. データ収集（対象世代）
    if not os.path.exists(f"data/horses_{target_year}.csv"):
        print(f"\n--- {target_year}年世代のデータを収集 ---")
        collect_data(target_year, max_horses=max_horses)
    else:
        print(f"\n--- {target_year}年世代: データ既存。スキップ ---")

    # 3. モデル学習
    predictor = train_model(training_years)

    # 4. 予測
    result = predict_top10(predictor, target_year, top_n=top_n)

    return result


def main():
    parser = argparse.ArgumentParser(description="POG予測システム")
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "predict", "full"],
        default="full",
        help="実行モード (default: full)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="対象の生年 (default: 2024)",
    )
    parser.add_argument(
        "--training-years",
        type=int,
        nargs="+",
        default=None,
        help="学習に使う世代の生年 (default: 対象年の前5年)",
    )
    parser.add_argument(
        "--max-horses",
        type=int,
        default=None,
        help="取得する最大馬数（テスト用）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="上位何頭を出力するか (default: 10)",
    )

    args = parser.parse_args()

    if args.mode == "collect":
        collect_data(args.year, max_horses=args.max_horses)
    elif args.mode == "train":
        years = args.training_years or list(range(args.year - 5, args.year))
        train_model(years)
    elif args.mode == "predict":
        predictor = POGPredictor()
        if os.path.exists("models/pog_predictor.pkl"):
            predictor.load()
        predict_top10(predictor, args.year, top_n=args.top_n)
    elif args.mode == "full":
        run_full_pipeline(
            target_year=args.year,
            training_years=args.training_years,
            max_horses=args.max_horses,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
