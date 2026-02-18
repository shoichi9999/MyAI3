"""
指定年の産駒データを一括取得するスクリプト。

取得項目:
  馬名, 性, 生年, 厩舎, 父, 母, 母父,
  父の年齢, 母の年齢, 母父の年齢, 馬主, 生産者

使い方:
  python scripts/fetch_progeny_data.py 2024
  python scripts/fetch_progeny_data.py 2024 --max-horses 100  # テスト用
"""

import argparse
import sys
import os
import json
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper import (
    fetch_horse_list_by_year,
    fetch_pedigree_birth_years,
    REQUEST_INTERVAL,
)

PEDIGREE_EMPTY = {
    "sire_birth_year": None,
    "dam_birth_year": None,
    "sire_sire_birth_year": None,
    "sire_dam_birth_year": None,
    "dam_sire_birth_year": None,
    "dam_dam_birth_year": None,
    "sire_sire_sire_birth_year": None,
    "sire_sire_dam_birth_year": None,
    "sire_dam_sire_birth_year": None,
    "sire_dam_dam_birth_year": None,
    "dam_sire_sire_birth_year": None,
    "dam_sire_dam_birth_year": None,
    "dam_dam_sire_birth_year": None,
    "dam_dam_dam_birth_year": None,
}


def fetch_progeny_data(birth_year: int, max_horses: int = None) -> pd.DataFrame:
    """
    指定年の産駒データを一括取得する。

    Parameters
    ----------
    birth_year : int
        対象の生年（例: 2024）
    max_horses : int, optional
        最大取得頭数（テスト用）。Noneで全頭取得。
    """
    # --- 馬一覧を取得 ---
    print(f"=== {birth_year}年生まれの産駒データを取得中 ===")
    max_pages = None if not max_horses else (max_horses // 100) + 1
    horse_list = fetch_horse_list_by_year(birth_year, max_pages=max_pages)

    if horse_list.empty:
        print("[WARN] 馬一覧の取得に失敗しました。")
        return pd.DataFrame()

    if max_horses:
        horse_list = horse_list.head(max_horses)

    print(f"  馬一覧取得完了: {len(horse_list)}頭")
    horse_list.to_csv(f"data/horses_{birth_year}.csv", index=False, encoding="utf-8-sig")

    # --- 血統ページから先祖の生年を取得 ---
    print(f"\n=== 血統データを取得中 ===")
    cache_path = f"data/parent_ages_{birth_year}.json"

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            if "sire_birth_year" in v:
                cache[k] = v
        print(f"  キャッシュ: {len(cache)}頭分あり")

    total = len(horse_list)
    errors = 0

    for i, (_, row) in enumerate(horse_list.iterrows()):
        hid = str(row["horse_id"])
        if hid in cache:
            continue

        try:
            time.sleep(REQUEST_INTERVAL)
            cache[hid] = fetch_pedigree_birth_years(hid)
        except Exception as e:
            cache[hid] = dict(PEDIGREE_EMPTY)
            errors += 1
            print(f"  [ERROR] {row['horse_name']}: {e}")

        if (i + 1) % 50 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            done = sum(1 for v in cache.values() if v.get("sire_birth_year"))
            print(f"  {i+1}/{total} 処理済み (成功: {done}, エラー: {errors})")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  血統データ取得完了 (エラー: {errors})")

    # --- 統合CSV構築 ---
    rows = []
    for _, horse in horse_list.iterrows():
        hid = str(horse["horse_id"])
        ped = cache.get(hid, {})

        sire_by = ped.get("sire_birth_year")
        dam_by = ped.get("dam_birth_year")
        dam_sire_by = ped.get("dam_sire_birth_year")

        rows.append({
            "馬名": horse.get("horse_name", ""),
            "性": horse.get("sex", ""),
            "生年": birth_year,
            "厩舎": horse.get("trainer", ""),
            "父": horse.get("sire", ""),
            "母": horse.get("dam", ""),
            "母父": horse.get("sire_of_dam", ""),
            "父の年齢": (birth_year - sire_by) if sire_by else None,
            "母の年齢": (birth_year - dam_by) if dam_by else None,
            "母父の年齢": (birth_year - dam_sire_by) if dam_sire_by else None,
            "馬主": horse.get("owner", ""),
            "生産者": horse.get("breeder", ""),
        })

    result_df = pd.DataFrame(rows)
    output_path = f"data/progeny_{birth_year}.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 完了: {len(result_df)}頭 → {output_path} ===")
    print(f"  性別内訳: {result_df['性'].value_counts().to_dict()}")
    for col in ["父の年齢", "母の年齢", "母父の年齢"]:
        valid = result_df[col].dropna()
        if len(valid) > 0:
            print(f"  {col}: 平均{valid.mean():.1f}歳 ({valid.min():.0f}-{valid.max():.0f})")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定年の産駒データを一括取得")
    parser.add_argument("year", type=int, help="対象の生年（例: 2024）")
    parser.add_argument("--max-horses", type=int, default=None, help="最大取得頭数（テスト用）")
    args = parser.parse_args()
    fetch_progeny_data(args.year, max_horses=args.max_horses)
