"""
セリ取引価格と産駒番号（何番仔か）を一括取得するスクリプト。

使い方:
  python scripts/fetch_extra_features.py 2024
  python scripts/fetch_extra_features.py 2024 --max-horses 50  # テスト用
"""

import argparse
import sys
import os
import json
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper import (
    fetch_horse_extra,
    fetch_dam_foal_list,
    REQUEST_INTERVAL,
)


def fetch_extra_features(birth_year: int, max_horses: int = None):
    """
    指定年の全馬について、セリ価格・母馬ID・何番仔かを取得する。

    出力: data/extra_features_{year}.json
    """
    csv_path = f"data/horses_{birth_year}.csv"
    cache_path = f"data/extra_features_{birth_year}.json"

    horses = pd.read_csv(csv_path)
    if max_horses:
        horses = horses.head(max_horses)
    print(f"{birth_year}年世代: {len(horses)}頭")

    # キャッシュ読み込み
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  キャッシュ: {len(cache)}頭分あり")

    # --- Phase 1: プロフィールページからセリ価格+母馬IDを取得 ---
    print(f"\n=== Phase 1: セリ価格・母馬ID取得 ===")
    total = len(horses)
    errors = 0

    for i, (_, row) in enumerate(horses.iterrows()):
        hid = str(row["horse_id"])
        if hid in cache and "sale_price" in cache[hid]:
            continue

        try:
            time.sleep(REQUEST_INTERVAL)
            extra = fetch_horse_extra(hid)
            if hid not in cache:
                cache[hid] = {}
            cache[hid]["sale_price"] = extra["sale_price"]
            cache[hid]["dam_id"] = extra["dam_id"]
        except Exception as e:
            if hid not in cache:
                cache[hid] = {}
            cache[hid].setdefault("sale_price", None)
            cache[hid].setdefault("dam_id", None)
            errors += 1
            print(f"  [ERROR] {row['horse_name']}: {e}")

        if (i + 1) % 50 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            has_price = sum(1 for v in cache.values() if v.get("sale_price"))
            print(f"  {i+1}/{total} (セリ価格あり: {has_price}, エラー: {errors})")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    has_price = sum(1 for v in cache.values() if v.get("sale_price"))
    print(f"  Phase 1完了: セリ価格あり {has_price}/{total}頭")

    # --- Phase 2: 母馬ページから産駒番号を取得 ---
    print(f"\n=== Phase 2: 産駒番号（何番仔か）取得 ===")

    # 未取得の馬の母馬IDを収集（ユニークなdam_idごとに1回だけアクセス）
    dam_foals_cache: dict[str, list[str]] = {}
    dams_to_fetch = set()

    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        entry = cache.get(hid, {})
        dam_id = entry.get("dam_id")
        if dam_id and "foal_number" not in entry:
            dams_to_fetch.add(dam_id)

    print(f"  取得対象の母馬: {len(dams_to_fetch)}頭")
    errors2 = 0

    for i, dam_id in enumerate(dams_to_fetch):
        try:
            time.sleep(REQUEST_INTERVAL)
            foal_list = fetch_dam_foal_list(dam_id)
            dam_foals_cache[dam_id] = foal_list
        except Exception as e:
            dam_foals_cache[dam_id] = []
            errors2 += 1
            print(f"  [ERROR] dam={dam_id}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dams_to_fetch)} 母馬処理済み (エラー: {errors2})")

    # 産駒番号を算出してキャッシュに書き込み
    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        entry = cache.get(hid, {})
        dam_id = entry.get("dam_id")
        if dam_id and "foal_number" not in entry:
            foal_list = dam_foals_cache.get(dam_id, [])
            if hid in foal_list:
                entry["foal_number"] = foal_list.index(hid) + 1
            else:
                entry["foal_number"] = None
            cache[hid] = entry

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    has_foal = sum(1 for v in cache.values() if v.get("foal_number"))
    print(f"  Phase 2完了: 産駒番号あり {has_foal}/{total}頭")

    # --- サマリー ---
    print(f"\n=== 完了 → {cache_path} ===")
    prices = [v["sale_price"] for v in cache.values() if v.get("sale_price")]
    foals = [v["foal_number"] for v in cache.values() if v.get("foal_number")]
    if prices:
        print(f"  セリ価格: 中央値{sorted(prices)[len(prices)//2]:.0f}万円 "
              f"({len(prices)}/{total}頭)")
    if foals:
        import statistics
        print(f"  産駒番号: 平均{statistics.mean(foals):.1f}番仔 "
              f"({len(foals)}/{total}頭)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="セリ価格・産駒番号を取得")
    parser.add_argument("year", type=int, help="対象の生年")
    parser.add_argument("--max-horses", type=int, default=None, help="最大取得頭数（テスト用）")
    args = parser.parse_args()
    fetch_extra_features(args.year, max_horses=args.max_horses)
