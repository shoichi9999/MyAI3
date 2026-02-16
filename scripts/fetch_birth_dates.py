"""指定世代の全馬の生年月日をスクレイピングしてJSONに保存する。"""

import sys
import os
import json
import re
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_INTERVAL = 1.0


def fetch_birth_date(horse_id: str) -> str | None:
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "lxml")
    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table:
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "生年月日" in th.text:
                return td.text.strip()
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_birth_dates.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])
    csv_path = f"data/horses_{year}.csv"
    cache_path = f"data/birth_dates_{year}.json"

    horses = pd.read_csv(csv_path)
    print(f"{year}年世代: {len(horses)}頭")

    # 既存キャッシュを読み込み
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"キャッシュ: {len(cache)}頭分あり")

    total = len(horses)
    errors = 0

    for i, (_, row) in enumerate(horses.iterrows()):
        hid = str(row["horse_id"])
        if hid in cache:
            continue

        try:
            time.sleep(REQUEST_INTERVAL)
            bd = fetch_birth_date(hid)
            if bd:
                cache[hid] = bd
            else:
                cache[hid] = ""
                errors += 1
        except Exception as e:
            cache[hid] = ""
            errors += 1
            print(f"  [ERROR] {row['horse_name']}: {e}")

        if (i + 1) % 50 == 0:
            # 中間保存
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            done = sum(1 for v in cache.values() if v)
            print(f"  {i+1}/{total} 処理済み (取得成功: {done}, エラー: {errors})")

    # 最終保存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    done = sum(1 for v in cache.values() if v)
    print(f"\n完了: {done}/{total}頭の生年月日を取得 (エラー: {errors})")


if __name__ == "__main__":
    main()
