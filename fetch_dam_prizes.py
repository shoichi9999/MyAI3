"""母馬の獲得賞金を一括取得するスクリプト。"""
import json
import os
import re
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from src.scraper import _rate_limited_sleep, _session, BASE_URL, concurrent_fetch

CACHE_FILE = "data/dam_prizes.json"


def parse_prize_text(text: str) -> float:
    """賞金テキスト(例: '1億4,545万円')を万円単位のfloatに変換。"""
    if not text:
        return 0.0
    text = text.replace(" ", "").replace(",", "").replace("円", "")
    total = 0.0
    m = re.search(r"(\d+)億", text)
    if m:
        total += int(m.group(1)) * 10000
    m = re.search(r"(\d+)万", text)
    if m:
        total += int(m.group(1))
    return total


def _parse_prize_from_soup(soup) -> float:
    """BeautifulSoupオブジェクトから獲得賞金(万円)を抽出する。"""
    prof_table = soup.find("table", class_="db_prof_table")
    if not prof_table:
        return 0.0
    # 中央賞金を優先
    for row in prof_table.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if th and td and "賞金" in th.text and "中央" in th.text:
            return parse_prize_text(td.text)
    # フォールバック: any 賞金
    for row in prof_table.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if th and td and "賞金" in th.text:
            return parse_prize_text(td.text)
    return 0.0


def _parse_foals_from_mare_soup(soup, dam_id: str) -> list[str]:
    """繁殖牝馬ページ (/horse/mare/{id}/) のsoupから産駒horse_idリストを抽出する。"""
    table = soup.find("table", class_="nk_tb_common")
    if not table:
        return []
    foal_ids = []
    for row in table.find_all("tr")[1:]:
        for a_tag in row.find_all("a"):
            href = a_tag.get("href", "")
            m = re.search(r"/horse/(\w+)/", href)
            if m and m.group(1) != dam_id:
                foal_ids.append(m.group(1))
                break
    return foal_ids


def fetch_dam_prize_and_foals(dam_id: str) -> dict:
    """母馬の賞金（プロフィールページ）と産駒リスト（繁殖牝馬ページ）を取得する。

    2リクエストを並列発行して高速化。

    Returns {"prize": float, "foals": list[str]}
    """
    if not dam_id or not str(dam_id).strip():
        return {"prize": 0.0, "foals": []}

    from bs4 import BeautifulSoup

    def _fetch_prize():
        _rate_limited_sleep()
        try:
            resp = _session.get(f"{BASE_URL}/horse/{dam_id}/", timeout=30)
            resp.encoding = "EUC-JP"
            soup = BeautifulSoup(resp.text, "lxml")
            return _parse_prize_from_soup(soup)
        except Exception as e:
            print(f"  [WARN] {dam_id} prize: {e}")
            return 0.0

    def _fetch_foals():
        _rate_limited_sleep()
        try:
            resp = _session.get(f"{BASE_URL}/horse/mare/{dam_id}/", timeout=30)
            resp.encoding = "EUC-JP"
            soup = BeautifulSoup(resp.text, "lxml")
            return _parse_foals_from_mare_soup(soup, dam_id)
        except Exception as e:
            print(f"  [WARN] {dam_id} foals: {e}")
            return []

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_prize = ex.submit(_fetch_prize)
        f_foals = ex.submit(_fetch_foals)
        prize = f_prize.result()
        foals = f_foals.result()

    return {"prize": prize, "foals": foals}


def fetch_horse_prize_by_id(horse_id: str) -> float:
    """horse_idのプロフィールページから獲得賞金(万円)を直接取得する。"""
    if not horse_id or not str(horse_id).strip():
        return 0.0

    _rate_limited_sleep()
    url = f"{BASE_URL}/horse/{horse_id}/"

    try:
        resp = _session.get(url, timeout=30)
        resp.encoding = "EUC-JP"
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  [WARN] {horse_id}: {e}")
        return 0.0

    return _parse_prize_from_soup(soup)


def fetch_horse_prize_by_name(name: str) -> float:
    """netkeiba の名前検索で馬の獲得賞金(万円)を取得する。"""
    if not name or name.strip() == "":
        return 0.0

    _rate_limited_sleep()
    encoded = urllib.parse.quote(name.strip())
    url = f"{BASE_URL}/?pid=horse_list&word={encoded}&sort=prize&list=100"

    try:
        resp = _session.get(url, timeout=30)
        resp.encoding = "EUC-JP"
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  [WARN] {name}: {e}")
        return 0.0

    # パターン1: プロフィールページに直接遷移
    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table:
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "賞金" in th.text and "中央" in th.text:
                return parse_prize_text(td.text)
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "賞金" in th.text:
                return parse_prize_text(td.text)

    # パターン2: 一覧テーブル
    list_table = soup.find("table", class_="nk_tb_common")
    if list_table:
        rows = list_table.find_all("tr")[1:]
        best_prize = 0.0
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 12:
                horse_name = cols[1].text.strip()
                if horse_name == name:
                    prize_text = cols[11].text.strip()
                    try:
                        prize = float(prize_text.replace(",", ""))
                    except ValueError:
                        prize = 0.0
                    best_prize = max(best_prize, prize)
        return best_prize

    return 0.0


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="母馬獲得賞金の一括取得")
    parser.add_argument("--year", type=int, default=None,
                        help="対象の生年（省略時は全CSVから母馬を収集）")
    args = parser.parse_args()

    if args.year:
        csv_files = [f"data/horses_{args.year}.csv"]
    else:
        csv_files = sorted(glob.glob("data/horses_*.csv"))
        csv_files = [f for f in csv_files if "_bak" not in f]

    all_dams = set()
    dam_id_map = {}  # {dam_name: dam_id}
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"[WARN] {csv_file} が見つかりません")
            continue
        horses = pd.read_csv(csv_file)
        dams = horses["dam"].dropna().unique()
        all_dams.update(d for d in dams if d.strip())
        # dam_name → dam_id マッピング構築
        if "dam_id" in horses.columns:
            for _, row in horses.iterrows():
                name = row.get("dam")
                did = row.get("dam_id")
                if pd.notna(name) and name.strip() and pd.notna(did) and str(did).strip():
                    dam_id_map[name.strip()] = str(did).strip()
        print(f"  {csv_file}: {len(dams)}頭の母馬")

    dams = sorted(all_dams)
    print(f"  dam_id 判明: {len(dam_id_map)}頭（直接アクセスで高速取得）")
    print(f"ユニーク母馬: {len(dams)}頭")

    # キャッシュ読み込み
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    to_fetch = [d for d in dams if d not in cache]
    print(f"キャッシュ済: {len(dams) - len(to_fetch)}頭, 新規: {len(to_fetch)}頭")

    if to_fetch:
        cache_lock = threading.Lock()

        def _fetch_prize(name):
            did = dam_id_map.get(name)
            if did:
                return fetch_horse_prize_by_id(did)
            return fetch_horse_prize_by_name(name)

        def on_prize_result(name, prize, _idx):
            with cache_lock:
                cache[name] = prize if prize is not None else 0.0

        def save_prizes():
            with cache_lock:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(dict(cache), f, ensure_ascii=False, indent=2)

        concurrent_fetch(
            items=to_fetch,
            fetch_fn=_fetch_prize,
            label="母馬賞金",
            on_result=on_prize_result,
            save_interval=100,
            save_fn=save_prizes,
        )

    print(f"\n完了: {len(cache)}頭の母馬賞金を保存")

    # 統計
    prizes = [v for v in cache.values() if v > 0]
    print(f"賞金 > 0の母馬: {len(prizes)}頭 ({len(prizes)/len(cache)*100:.1f}%)")
    if prizes:
        import numpy as np
        print(f"平均: {np.mean(prizes):,.0f}万  中央値: {np.median(prizes):,.0f}万  最大: {max(prizes):,.0f}万")


if __name__ == "__main__":
    main()
