"""種牡馬の競走時の距離適性データを netkeibaから取得する。

各馬ページの「競走成績」から G1〜G3 で勝った距離を抽出し、
最大/最頻距離を判定材料にする。
"""
import sys, os, json, time, re
sys.path.insert(0, ".")
from src.scraper import _get_soup

CACHE = "data/sire_distance.json"


def fetch_winning_distances(horse_id: str) -> dict | None:
    """馬の競走成績から「1着になったレースの距離」と「重賞勝ちレース」を取得。"""
    try:
        url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
        soup = _get_soup(url)
        # 競走成績テーブル
        table = soup.find("table", class_="db_h_race_results")
        if not table:
            return {"wins": [], "grade_wins": []}

        wins = []  # 全勝利の距離
        grade_wins = []  # 重賞勝ちのリスト [{race, distance, grade}]

        rows = table.find_all("tr")
        if len(rows) < 2:
            return {"wins": [], "grade_wins": []}

        # ヘッダから列インデックスを特定
        header = rows[0].find_all(["th", "td"])
        headers = [h.get_text(strip=True) for h in header]
        try:
            idx_race = headers.index("レース名")
        except ValueError:
            idx_race = 4
        try:
            idx_rank = headers.index("着順")
        except ValueError:
            idx_rank = 11
        try:
            idx_dist = headers.index("距離")
        except ValueError:
            idx_dist = 14

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) <= max(idx_race, idx_rank, idx_dist):
                continue
            rank = cells[idx_rank].get_text(strip=True)
            if rank != "1":
                continue
            race_name = cells[idx_race].get_text(strip=True)
            dist_text = cells[idx_dist].get_text(strip=True)
            m = re.search(r"(\d{3,4})", dist_text)
            if not m:
                continue
            distance = int(m.group(1))
            wins.append(distance)
            # 重賞判定: レース名や aタグの class、または icon で判定
            race_link = cells[idx_race].find("a")
            race_html = str(cells[idx_race])
            grade = None
            if "GⅠ" in race_html or "G1" in race_html or "(G1)" in race_name or "GI" in race_html:
                grade = "G1"
            elif "GⅡ" in race_html or "G2" in race_html or "GII" in race_html:
                grade = "G2"
            elif "GⅢ" in race_html or "G3" in race_html or "GIII" in race_html:
                grade = "G3"
            if grade:
                grade_wins.append({"race": race_name, "distance": distance, "grade": grade})

        return {"wins": wins, "grade_wins": grade_wins}
    except Exception as e:
        return None


def main():
    import pandas as pd

    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding="utf-8"))

    # 対象 sire_id, bms_id を集める
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
        info = fetch_winning_distances(hid)
        if info is not None:
            cache[hid] = info
        if i % 50 == 0:
            mx = max(info["wins"]) if (info and info["wins"]) else 0
            ng = len(info["grade_wins"]) if info else 0
            print(f"  {i}/{len(to_fetch)}  {hid} -> max_dist={mx}, graded_wins={ng}")
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        time.sleep(1.0)  # netkeibaへの負荷配慮

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"完了: 計 {len(cache)}頭の戦績取得済み")

    # 距離分布集計
    from collections import Counter
    bucket = Counter()
    for s in cache.values():
        if not s or not s.get("wins"):
            bucket["NO_WIN"] += 1
            continue
        mx = max(s["wins"])
        if mx >= 2800: bucket["STAYER(2800+)"] += 1
        elif mx >= 2400: bucket["MID_DIST(2400-2799)"] += 1
        elif mx >= 1800: bucket["MILER(1800-2399)"] += 1
        elif mx >= 1400: bucket["MILER(1400-1799)"] += 1
        else: bucket["SPRINTER(<1400)"] += 1
    print("\n=== 最大勝ち距離による分類 ===")
    for k, n in bucket.most_common():
        print(f"  {k}: {n}頭")


if __name__ == "__main__":
    main()
