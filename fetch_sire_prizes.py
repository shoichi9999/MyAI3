"""種牡馬自身の現役時代の獲得賞金を一括取得するスクリプト。

初年度種牡馬のスコアリング補正に使用する。
"""
import glob
import json
import os
import threading

import pandas as pd
from fetch_dam_prizes import fetch_horse_prize_by_id
from src.scraper import concurrent_fetch

CACHE_FILE = "data/sire_prizes.json"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="種牡馬の現役獲得賞金を一括取得")
    parser.add_argument("--year", type=int, default=None,
                        help="対象の生年（省略時は全CSVから種牡馬を収集）")
    args = parser.parse_args()

    if args.year:
        csv_files = [f"data/horses_{args.year}.csv"]
    else:
        csv_files = sorted(glob.glob("data/horses_*.csv"))
        csv_files = [f for f in csv_files if "_bak" not in f]

    sire_id_map = {}  # {sire_name: sire_id}
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"[WARN] {csv_file} が見つかりません")
            continue
        horses = pd.read_csv(csv_file)
        if "sire_id" in horses.columns:
            for _, row in horses.iterrows():
                name = row.get("sire")
                sid = row.get("sire_id")
                if pd.notna(name) and str(name).strip() and pd.notna(sid) and str(sid).strip():
                    sire_id_map[str(name).strip()] = str(sid).strip()
        print(f"  {csv_file}: {horses['sire'].nunique()}頭の種牡馬")

    sires = sorted(sire_id_map.keys())
    print(f"ユニーク種牡馬: {len(sires)}頭（全馬sire_idあり）")

    # キャッシュ読み込み
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    to_fetch = [s for s in sires if s not in cache]
    print(f"キャッシュ済: {len(sires) - len(to_fetch)}頭, 新規: {len(to_fetch)}頭")

    if to_fetch:
        cache_lock = threading.Lock()

        def _fetch_prize(name):
            sid = sire_id_map[name]
            return fetch_horse_prize_by_id(sid)

        def on_result(name, prize, _idx):
            with cache_lock:
                cache[name] = prize if prize is not None else 0.0

        def save_cache():
            with cache_lock:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(dict(cache), f, ensure_ascii=False, indent=2)

        concurrent_fetch(
            items=to_fetch,
            fetch_fn=_fetch_prize,
            label="種牡馬賞金",
            on_result=on_result,
            save_interval=50,
            save_fn=save_cache,
        )

    # 最終保存
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(cache)}頭の種牡馬賞金を保存")

    # 統計
    prizes = [v for v in cache.values() if v > 0]
    print(f"賞金 > 0の種牡馬: {len(prizes)}頭 ({len(prizes)/len(cache)*100:.1f}%)")
    if prizes:
        import numpy as np
        print(f"平均: {np.mean(prizes):,.0f}万  中央値: {np.median(prizes):,.0f}万  最大: {max(prizes):,.0f}万")


if __name__ == "__main__":
    main()
