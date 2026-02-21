"""母馬の賞金+産駒リストを一括取得するパイプライン。"""
import json
import os
import sys
import threading

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from fetch_dam_prizes import fetch_dam_prize_and_foals
from src.scraper import concurrent_fetch

PRIZE_FILE = "data/dam_prizes.json"
FOALS_FILE = "data/dam_foals.json"


def _load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 全CSVから母馬を収集
    all_dams = {}  # {dam_id: dam_name}
    for year in [2019, 2024]:
        csv_path = f"data/horses_{year}.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            name = row.get("dam")
            did = row.get("dam_id")
            if pd.notna(name) and pd.notna(did):
                all_dams[str(did).strip()] = name.strip()

    print(f"ユニーク母馬: {len(all_dams)}頭")

    # キャッシュ読み込み
    prize_cache = _load(PRIZE_FILE)
    foals_cache = _load(FOALS_FILE)

    # 未取得の母馬を抽出（prizeまたはfoalsが未取得）
    to_fetch = []
    for did, name in all_dams.items():
        if name not in prize_cache or did not in foals_cache:
            to_fetch.append(did)

    print(f"キャッシュ済: prize={len(prize_cache)}, foals={len(foals_cache)}")
    print(f"新規取得: {len(to_fetch)}頭")

    if not to_fetch:
        print("全てキャッシュ済み")
        return

    lock = threading.Lock()

    def on_result(dam_id, result, _idx):
        with lock:
            if result is not None:
                name = all_dams.get(dam_id, "")
                if name:
                    prize_cache[name] = result["prize"]
                foals_cache[dam_id] = result["foals"]
            else:
                name = all_dams.get(dam_id, "")
                if name and name not in prize_cache:
                    prize_cache[name] = 0.0
                if dam_id not in foals_cache:
                    foals_cache[dam_id] = []

    def save_fn():
        with lock:
            _save(PRIZE_FILE, dict(prize_cache))
            _save(FOALS_FILE, dict(foals_cache))

    concurrent_fetch(
        items=to_fetch,
        fetch_fn=fetch_dam_prize_and_foals,
        label="母馬賞金+産駒",
        on_result=on_result,
        save_interval=100,
        save_fn=save_fn,
    )

    print(f"\n完了: prize={len(prize_cache)}, foals={len(foals_cache)}")


if __name__ == "__main__":
    main()
