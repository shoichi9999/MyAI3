"""
追加特徴量データを一括取得するスクリプト。

馬一覧CSV取得後に実行し、以下を補完する:
  - 生年月日 → birth_dates_{year}.json
  - セリ価格・産駒番号 → extra_features_{year}.json

※ 親の生年（父・母・母父）は馬一覧取得時にCSVへ保存済みのため不要。

使い方:
  python scripts/fetch_all_features.py 2024
  python scripts/fetch_all_features.py 2024 --max-horses 100
"""

import argparse
import sys
import os
import json
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper import (
    fetch_horse_profile,
    fetch_dam_foal_list,
    REQUEST_INTERVAL,
)


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_all_features(birth_year: int, max_horses: int = None):
    csv_path = f"data/horses_{birth_year}.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} が見つかりません。先にデータを収集してください。")
        return

    horses = pd.read_csv(csv_path)
    if max_horses:
        horses = horses.head(max_horses)
    total = len(horses)
    print(f"=== {birth_year}年世代: {total}頭 ===\n")

    # キャッシュファイル
    bd_path = f"data/birth_dates_{birth_year}.json"
    ef_path = f"data/extra_features_{birth_year}.json"

    bd_cache = _load_cache(bd_path)    # {horse_id: "YYYY年MM月DD日"}
    ef_cache = _load_cache(ef_path)    # {horse_id: {sale_price: N, foal_number: N}}

    print(f"  既存キャッシュ: birth_dates={len(bd_cache)}, extra_features={len(ef_cache)}")

    # --- Phase 1: プロフィールページ（生年月日 + セリ価格） ---
    print(f"\n=== Phase 1: プロフィールデータ取得 ===")
    errors = 0

    for i, (_, row) in enumerate(horses.iterrows()):
        hid = str(row["horse_id"])

        has_bd = hid in bd_cache
        has_ef = hid in ef_cache and "sale_price" in ef_cache.get(hid, {})
        if has_bd and has_ef:
            continue

        try:
            profile = fetch_horse_profile(hid)
            bd_cache[hid] = profile["birth_date"] or ""
            if hid not in ef_cache:
                ef_cache[hid] = {}
            ef_cache[hid]["sale_price"] = profile["sale_price"]
        except Exception as e:
            bd_cache[hid] = ""
            if hid not in ef_cache:
                ef_cache[hid] = {}
            ef_cache[hid].setdefault("sale_price", None)
            errors += 1
            print(f"  [ERROR] {row['horse_name']}: {e}")

        if (i + 1) % 50 == 0:
            _save_cache(bd_path, bd_cache)
            _save_cache(ef_path, ef_cache)
            print(f"  {i+1}/{total} 処理済み (エラー: {errors})")

    _save_cache(bd_path, bd_cache)
    _save_cache(ef_path, ef_cache)
    print(f"  Phase 1完了 (エラー: {errors})")

    # --- Phase 2: 産駒番号（母馬ページから何番仔かを取得） ---
    print(f"\n=== Phase 2: 産駒番号（何番仔か）取得 ===")

    # CSVのdam_idカラムを使用（馬一覧取得時に保存済み）
    dams_to_fetch = set()
    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        entry = ef_cache.get(hid, {})
        dam_id = str(row.get("dam_id", "")) if pd.notna(row.get("dam_id")) else ""
        if dam_id and "foal_number" not in entry:
            dams_to_fetch.add(dam_id)
            if hid not in ef_cache:
                ef_cache[hid] = {}
            ef_cache[hid]["dam_id"] = dam_id

    print(f"  未取得の母馬: {len(dams_to_fetch)}頭")
    dam_foals = {}
    errors2 = 0

    for i, dam_id in enumerate(dams_to_fetch):
        try:
            time.sleep(REQUEST_INTERVAL)
            dam_foals[dam_id] = fetch_dam_foal_list(dam_id)
        except Exception as e:
            dam_foals[dam_id] = []
            errors2 += 1
            print(f"  [ERROR] dam={dam_id}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dams_to_fetch)} 母馬処理済み")

    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        entry = ef_cache.get(hid, {})
        dam_id = entry.get("dam_id") or (str(row.get("dam_id", "")) if pd.notna(row.get("dam_id")) else "")
        if dam_id and "foal_number" not in entry:
            foal_list = dam_foals.get(dam_id, [])
            entry["foal_number"] = (foal_list.index(hid) + 1) if hid in foal_list else None
            ef_cache[hid] = entry

    _save_cache(ef_path, ef_cache)
    print(f"  Phase 2完了 (エラー: {errors2})")

    # --- サマリー ---
    print(f"\n=== 完了 ===")
    bd_ok = sum(1 for v in bd_cache.values() if v)
    ef_price = sum(1 for v in ef_cache.values() if v.get("sale_price"))
    ef_foal = sum(1 for v in ef_cache.values() if v.get("foal_number"))
    print(f"  生年月日:  {bd_ok}/{total}")
    print(f"  セリ価格:  {ef_price}/{total}")
    print(f"  産駒番号:  {ef_foal}/{total}")
    print(f"  → {bd_path}, {ef_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="追加特徴量データを一括取得")
    parser.add_argument("year", type=int, help="対象の生年（例: 2024）")
    parser.add_argument("--max-horses", type=int, default=None, help="最大取得頭数")
    args = parser.parse_args()
    fetch_all_features(args.year, max_horses=args.max_horses)
