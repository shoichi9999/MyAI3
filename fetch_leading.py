"""
リーディングサイアー / BMSリーディングデータを netkeiba からスクレイピング。

使い方:
  python fetch_leading.py 2020 2024 2025
"""

import json
import os
import sys
import time

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DELAY = 1.5  # リクエスト間隔（秒）


def fetch_leading_page(kind: str, year: int, page: int, max_retries: int = 4) -> list[dict]:
    """1ページ分のリーディングデータを取得する。

    Parameters
    ----------
    kind : str
        "sire_leading" or "bms_leading"
    year : int
    page : int
    max_retries : int
        リトライ回数（指数バックオフ）

    Returns
    -------
    list[dict]
        各種牡馬の情報 [{name, horse_id, rank, runners, winners,
                         win_rate, ei, prize, representative}, ...]
    """
    pid = kind  # sire_leading / bms_leading
    url = f"https://db.netkeiba.com/?pid={pid}&year={year}&page={page}"

    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [WARN] 接続エラー (attempt {attempt+1}/{max_retries}): {e} — {wait}秒待機")
                time.sleep(wait)
                continue
            print(f"    [ERROR] 接続エラー (最終): {e}")
            return []

        if resp.status_code == 200:
            resp.encoding = "euc-jp"
            soup = BeautifulSoup(resp.text, "html.parser")
            break
        else:
            wait = 2 ** (attempt + 1)
            print(f"    [WARN] HTTP {resp.status_code} (attempt {attempt+1}/{max_retries}): {url[:80]}... — {wait}秒待機")
            if attempt < max_retries - 1:
                time.sleep(wait)
                session = requests.Session()
                session.headers.update(HEADERS)
            else:
                print(f"    [ERROR] {max_retries}回リトライ後も失敗: HTTP {resp.status_code}")
                resp.encoding = "euc-jp"
                soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="nk_tb_common")
    if not table:
        return []

    rows = table.find_all("tr")
    # rows[0]: ヘッダー1, rows[1]: ヘッダー2, rows[2:]: データ
    results = []
    for row in rows[2:]:
        cells = row.find_all("td")
        if len(cells) < 19:
            continue

        # 馬名リンクから horse_id を抽出
        link = cells[1].find("a")
        href = link["href"] if link else ""
        # href例: /horse/sire/2010105827/
        horse_id = ""
        if href:
            parts = href.strip("/").split("/")
            horse_id = parts[-1] if parts else ""

        name = cells[1].get_text(strip=True)

        def _num(s: str) -> float:
            try:
                return float(s.replace(",", ""))
            except (ValueError, TypeError):
                return 0.0

        results.append({
            "name": name,
            "horse_id": horse_id,
            "rank": int(_num(cells[0].get_text(strip=True))),
            "runners": int(_num(cells[2].get_text(strip=True))),
            "winners": int(_num(cells[3].get_text(strip=True))),
            "win_rate": _num(cells[16].get_text(strip=True)),
            "ei": _num(cells[17].get_text(strip=True)),
            "progeny_prize": _num(cells[18].get_text(strip=True)),
            "representative": cells[21].get_text(strip=True) if len(cells) > 21 else "",
        })

    return results


def fetch_all_leading(kind: str, year: int, max_pages: int = 20) -> dict:
    """全ページをスクレイピングし、{馬名: {...}} 形式の辞書を返す。"""
    all_data = {}
    for page in range(1, max_pages + 1):
        rows = fetch_leading_page(kind, year, page)
        if not rows:
            break
        for r in rows:
            all_data[r["name"]] = {
                "horse_id": r["horse_id"],
                "rank": r["rank"],
                "runners": r["runners"],
                "winners": r["winners"],
                "win_rate": r["win_rate"],
                "ei": r["ei"],
                "progeny_prize": r["progeny_prize"],
                "representative": r["representative"],
            }
        print(f"  {kind} {year} page {page}: {len(rows)}件 (累計 {len(all_data)}件)")
        if len(rows) < 50:
            break
        time.sleep(DELAY)

    return all_data


def main():
    if len(sys.argv) < 2:
        print("使い方: python fetch_leading.py 2020 2024 2025")
        sys.exit(1)

    years = [int(y) for y in sys.argv[1:]]

    for year in years:
        for kind in ["sire_leading", "bms_leading"]:
            out_path = f"data/{kind}_{year}.json"

            if os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
                print(f"スキップ: {out_path} (既存 {len(existing)}件)")
                continue

            print(f"\n取得中: {kind} {year}")
            data = fetch_all_leading(kind, year, max_pages=20)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            ei_vals = [v["ei"] for v in data.values() if v["ei"] > 0]
            print(f"  保存: {out_path} ({len(data)}件, EI>0: {len(ei_vals)}件)")
            if ei_vals:
                print(f"  EI: 最大={max(ei_vals):.2f} 平均={sum(ei_vals)/len(ei_vals):.2f}")
            time.sleep(DELAY)

    print("\n完了")


if __name__ == "__main__":
    main()
