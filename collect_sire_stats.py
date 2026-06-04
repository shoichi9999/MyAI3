"""種牡馬の累計成績（重賞勝ち数・平均距離）をnetkeibaから取得。"""
import sys, os, json, time, re
sys.path.insert(0, ".")
from src.scraper import _get_soup

CACHE = "data/sire_stats.json"


def fetch_stats(horse_id: str) -> dict | None:
    """sire ページから累計の重賞・平均距離を取得（父用と母父用の両方）。"""
    try:
        url = f"https://db.netkeiba.com/horse/sire/{horse_id}/"
        soup = _get_soup(url)
        tables = soup.find_all("table")
        result = {"sire": None, "bms": None}
        # 2つのテーブル：0=父として、1=母父として
        for idx, key in [(0, "sire"), (1, "bms")]:
            if len(tables) <= idx:
                continue
            t = tables[idx]
            rows = t.find_all("tr")
            # 累計行を探す
            for r in rows:
                cells = r.find_all("td")
                if cells and cells[0].get_text(strip=True) == "累計":
                    texts = [c.get_text(strip=True) for c in cells]
                    # ヘッダ順: 年度,順位,出走頭数,勝馬頭数,出走回数,勝利回数,重賞,特別,平場,芝,ダート,勝馬率,EI,入着賞金,平均距離芝,平均距離ダ,代表馬
                    # ※数値の途中にコンマ付きがあるので、texts要素は単純列挙だがcellsはセルごと
                    # cells配列で取り直し
                    def to_num(s):
                        s = s.replace(",", "").replace("%", "").replace("万", "").strip()
                        try: return float(s)
                        except: return None

                    try:
                        graded = int(cells[6].get_text(strip=True))
                    except: graded = 0
                    avg_dist_turf = to_num(cells[14].get_text(strip=True)) or 0
                    avg_dist_dirt = to_num(cells[15].get_text(strip=True)) or 0
                    runners = to_num(cells[2].get_text(strip=True)) or 0
                    ei = to_num(cells[12].get_text(strip=True)) or 0
                    rep = cells[16].get_text(strip=True) if len(cells) > 16 else ""
                    result[key] = {
                        "graded": graded,
                        "avg_dist_turf": avg_dist_turf,
                        "avg_dist_dirt": avg_dist_dirt,
                        "runners": runners,
                        "ei": ei,
                        "rep": rep[:30],
                    }
                    break
        return result
    except Exception:
        return None


def main():
    import pandas as pd

    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding="utf-8"))

    targets = set()
    for y in range(2015, 2026):
        try:
            df = pd.read_csv(f"data/horses_{y}.csv")
            for col in ["sire_id", "bms_id"]:
                if col in df.columns:
                    vals = df[col].dropna().astype(str)
                    vals = vals[vals.str.match(r"^[0-9a-z]+$", na=False)]
                    targets.update(vals)
        except FileNotFoundError:
            pass

    to_fetch = [s for s in targets if s not in cache]
    print(f"対象: {len(targets)}頭、新規取得: {len(to_fetch)}頭")

    for i, hid in enumerate(to_fetch, 1):
        info = fetch_stats(hid)
        if info is not None:
            cache[hid] = info
        if i % 50 == 0:
            s = info["sire"] if info else None
            g = s["graded"] if s else "?"
            print(f"  {i}/{len(to_fetch)}  {hid} -> graded={g}")
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        time.sleep(1.0)

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"完了: {len(cache)}頭")

    # 重賞数分布
    from collections import Counter
    sire_graded = Counter()
    for v in cache.values():
        if not v or not v.get("sire"): continue
        g = v["sire"]["graded"]
        sire_graded["g0"] += 1 if g == 0 else 0
        sire_graded["g1-5"] += 1 if 1 <= g <= 5 else 0
        sire_graded["g6-20"] += 1 if 6 <= g <= 20 else 0
        sire_graded["g21+"] += 1 if g >= 21 else 0
    print("\n=== sire 重賞勝ち分布 ===")
    for k, n in sire_graded.most_common():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
