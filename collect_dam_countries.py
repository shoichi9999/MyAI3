"""輸入繁殖牝馬の産地を netkeibaから取得する。

netkeiba.com/horse/{dam_id}/ ページから生まれ国を抽出してキャッシュする。
"""
import sys, os, json, time, re
sys.path.insert(0, ".")
from src.scraper import _get_soup

CACHE = "data/dam_countries.json"


def fetch_country(dam_id: str) -> str | None:
    """指定 dam_id (000a始まり) の馬ページから産地コードを取得する。

    netkeibaのプロフィール表「産地: 亜」のような日本語1文字表記を抽出。
    """
    try:
        soup = _get_soup(f"https://db.netkeiba.com/horse/{dam_id}/")
        prof = soup.find("table", class_="db_prof_table")
        if prof:
            for row in prof.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    if label == "産地":
                        return cells[1].get_text(strip=True)
        return "UNK"
    except Exception:
        return None


def main():
    # ターゲット dam_id 収集
    import pandas as pd

    targets = set()
    # 過去9年の horses から imported_dam (000a始まり) の dam_id を集める
    for y in range(2015, 2026):
        try:
            df = pd.read_csv(f"data/horses_{y}.csv")
            ids = df[df["dam_id"].astype(str).str.startswith("000a", na=False)]["dam_id"].astype(str).unique()
            targets.update(ids)
        except FileNotFoundError:
            pass

    print(f"対象 dam_id: {len(targets)}頭")

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    to_fetch = [d for d in targets if d not in cache]
    print(f"新規取得: {len(to_fetch)}頭")

    for i, dam_id in enumerate(to_fetch, 1):
        country = fetch_country(dam_id)
        if country:
            cache[dam_id] = country
        if i % 50 == 0:
            print(f"  {i}/{len(to_fetch)}  {dam_id} -> {country}")
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        time.sleep(0.3)

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"完了: 計 {len(cache)}頭の産地取得済み")

    # 集計
    from collections import Counter
    c = Counter(cache.values())
    print("\n=== 産地分布 ===")
    for code, n in c.most_common():
        print(f"  {code}: {n}頭")


if __name__ == "__main__":
    main()
