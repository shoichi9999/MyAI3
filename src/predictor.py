"""
POG TOP10予測 - メイン実行モジュール。

使い方:
  1. データ収集モード: 対象年のデータをスクレイピング
  2. 予測モード: ヒューリスティックスコアでTOP10を予測
  3. 全自動モード: 上記すべてを一括実行
"""

import argparse
import os

import pandas as pd
from tabulate import tabulate

from src.scraper import fetch_horse_list_by_year
from src.features import build_feature_matrix
from src.model import heuristic_score


def collect_data(birth_year: int, max_horses: int = None) -> pd.DataFrame:
    """データを収集してCSVに保存する。"""
    print(f"\n{'='*60}")
    print(f"  データ収集: {birth_year}年生まれの2歳馬")
    print(f"{'='*60}")

    max_pages = None if not max_horses else (max_horses // 100) + 1
    horse_list = fetch_horse_list_by_year(birth_year, max_pages=max_pages)

    if horse_list.empty:
        print("[WARN] 馬一覧の取得に失敗しました。")
        return horse_list

    if max_horses:
        horse_list = horse_list.head(max_horses)

    print(f"  取得完了: {len(horse_list)}頭")
    horse_list.to_csv(f"data/horses_{birth_year}.csv", index=False, encoding="utf-8-sig")
    return horse_list


def predict_top(
    target_year: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    ヒューリスティックスコアでTOP N予測を行う。

    Parameters
    ----------
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
    features = build_feature_matrix(horses_df, birth_year=target_year)
    features["score"] = heuristic_score(features)

    # スコアでソートしてTOP N
    features = features.sort_values("score", ascending=False)
    top = features.head(top_n).copy()

    # 表示用に整形
    display_cols = [
        "horse_name", "score",
        "sire_ei", "bms_ei", "dam_prize",
        "birth_month", "sire_age", "dam_age",
    ]
    available_cols = [c for c in display_cols if c in top.columns]
    display_df = top[available_cols].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "順位"

    col_rename = {
        "horse_name": "馬名",
        "score": "スコア",
        "sire_ei": "父EI",
        "bms_ei": "母父EI",
        "dam_prize": "母馬賞金(万)",
        "birth_month": "生月",
        "sire_age": "父年齢",
        "dam_age": "母年齢",
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
    max_horses: int = None,
    top_n: int = 10,
):
    """
    全自動パイプライン（データ収集 → 予測）。

    Parameters
    ----------
    target_year : int
        予測対象の世代（生年）
    max_horses : int
        取得する最大馬数（テスト用）
    top_n : int
        上位何頭を出力するか
    """
    print("=" * 60)
    print("  POG予測システム - 全自動パイプライン")
    print(f"  対象世代: {target_year}年生まれ")
    print(f"  予測馬数: TOP{top_n}")
    print("=" * 60)

    # 1. データ収集
    if not os.path.exists(f"data/horses_{target_year}.csv"):
        collect_data(target_year, max_horses=max_horses)
    else:
        print(f"\n--- {target_year}年世代: データ既存。スキップ ---")

    # 2. 予測
    result = predict_top(target_year, top_n=top_n)

    return result


def main():
    parser = argparse.ArgumentParser(description="POG予測システム")
    parser.add_argument(
        "--mode",
        choices=["collect", "predict", "full"],
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
    elif args.mode == "predict":
        predict_top(args.year, top_n=args.top_n)
    elif args.mode == "full":
        run_full_pipeline(
            target_year=args.year,
            max_horses=args.max_horses,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
