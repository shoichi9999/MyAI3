"""
netkeiba.comから種牡馬リーディング・BMSリーディングを取得するスクリプト。

取得データ:
  - data/sire_leading_{year}.json  (種牡馬リーディング)
  - data/bms_leading_{year}.json   (母父馬リーディング)

使い方:
  python scripts/fetch_leading.py 2024
  python scripts/fetch_leading.py 2024 --type sire   # 種牡馬のみ
  python scripts/fetch_leading.py 2024 --type bms    # BMSのみ
  python scripts/fetch_leading.py --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper import _get_soup

BASE_URL = "https://db.netkeiba.com"


def _parse_number(text: str) -> float:
    """カンマ付き数値文字列をfloatに変換する。"""
    text = text.strip().replace(",", "")
    if not text or text == "-":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def fetch_leading_page(pid: str, year: int, page: int = 1) -> list[dict]:
    """リーディングページの1ページ分をパースする。

    Parameters
    ----------
    pid : str
        "sire_leading" or "bms_leading"
    year : int
        対象年度
    page : int
        ページ番号

    Returns
    -------
    list[dict]
        [{name, rank, progeny_prize, win_rate, ei}, ...]
    """
    url = f"{BASE_URL}/?pid={pid}&year={year}&page={page}"
    soup = _get_soup(url)

    table = soup.find("table", class_="nk_tb_common")
    if not table:
        return []

    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    # ヘッダー行からEI列のインデックスを特定
    header_row = rows[0]
    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

    # EI列を探す（"EI" or "ＥＩ" など）
    ei_idx = None
    prize_idx = None
    win_rate_idx = None
    for i, h in enumerate(headers):
        h_normalized = h.replace("Ｅ", "E").replace("Ｉ", "I")
        if h_normalized == "EI" or "アーニング" in h:
            ei_idx = i
        if "賞金" in h and prize_idx is None:
            prize_idx = i
        if "勝率" in h or "勝馬率" in h:
            win_rate_idx = i

    results = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        # 順位
        rank_text = cols[0].get_text(strip=True)
        rank = int(rank_text) if rank_text.isdigit() else 0

        # 種牡馬名（リンクテキスト）
        name_tag = cols[1].find("a")
        if not name_tag:
            continue
        name = name_tag.get_text(strip=True)
        if not name:
            continue

        entry = {"rank": rank}

        # 産駒賞金
        if prize_idx is not None and prize_idx < len(cols):
            entry["progeny_prize"] = _parse_number(cols[prize_idx].get_text())
        else:
            # フォールバック: 賞金は通常後半にある
            for i in range(len(cols) - 1, 3, -1):
                val = _parse_number(cols[i].get_text())
                if val > 100:  # 賞金は通常100万以上
                    entry["progeny_prize"] = val
                    break
            else:
                entry["progeny_prize"] = 0.0

        # EI
        if ei_idx is not None and ei_idx < len(cols):
            ei_text = cols[ei_idx].get_text(strip=True)
            entry["ei"] = ei_text if ei_text else "0.00"
        else:
            entry["ei"] = "0.00"

        # 勝率（sire_leadingのみ）
        if win_rate_idx is not None and win_rate_idx < len(cols):
            entry["win_rate"] = cols[win_rate_idx].get_text(strip=True)

        results.append((name, entry))

    return results


def fetch_leading(pid: str, year: int, max_pages: int = 10) -> dict:
    """リーディングデータを全ページ取得する。

    Returns
    -------
    dict
        {name: {rank, progeny_prize, ei, ...}, ...}
    """
    label = "種牡馬" if "sire" in pid else "BMS"
    print(f"\n=== {label}リーディング {year}年 ===")

    all_data = {}
    for page in range(1, max_pages + 1):
        entries = fetch_leading_page(pid, year, page)
        if not entries:
            break
        for name, data in entries:
            if name not in all_data:  # 重複排除
                all_data[name] = data
        print(f"  ページ {page}: {len(entries)}件取得 (累計: {len(all_data)})")
        if len(entries) < 20:  # 最終ページ
            break

    print(f"  合計: {len(all_data)}頭")
    return all_data


def save_leading(data: dict, path: str):
    """リーディングデータをJSONに保存する。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → {path}")


def main():
    parser = argparse.ArgumentParser(description="リーディングデータ取得")
    parser.add_argument("year", nargs="?", type=int, default=None,
                        help="対象年度（例: 2024）")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="複数年度を一括取得（例: --years 2015 2016 2017）")
    parser.add_argument("--type", choices=["sire", "bms", "both"], default="both",
                        help="取得タイプ（デフォルト: both）")
    args = parser.parse_args()

    if args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        parser.error("yearまたは--yearsを指定してください")

    for year in years:
        if args.type in ("sire", "both"):
            sire_data = fetch_leading("sire_leading", year)
            save_leading(sire_data, f"data/sire_leading_{year}.json")

        if args.type in ("bms", "both"):
            bms_data = fetch_leading("bms_leading", year)
            save_leading(bms_data, f"data/bms_leading_{year}.json")



if __name__ == "__main__":
    main()
