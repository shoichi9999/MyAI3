"""
B2: 補助重賞のrace_idを試行錯誤で特定し、TOP5 horse_idを取得する。

netkeibaのrace_id: YYYY + 場所(2桁) + 回数(2桁) + 日(2桁) + R(2桁)
場所コード: 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉
"""
import sys, json, time, os
sys.path.insert(0, ".")
from src.scraper import _get_soup

CACHE_RID = "data/aux_race_ids.json"
CACHE_TOP5 = "data/aux_race_top5.json"

# 各重賞の典型的race_idパターン（場所, 回数, 日, R）
# 年により振替・場所変更があるので近傍も探索
RACE_PATTERNS = {
    "satsuki":   {"places": ["06"], "kai": ["03"], "day_range": (6, 10), "R": "11"},  # 皐月賞: 中山3回
    "kikuka":    {"places": ["08", "09"], "kai": ["04", "05"], "day_range": (6, 10), "R": "11"},  # 菊花賞: 京都4回（阪神振替あり）
    "sakura":    {"places": ["09", "08"], "kai": ["02", "03"], "day_range": (4, 8), "R": "11"},  # 桜花賞: 阪神2回（京都振替あり）
    "syuka":     {"places": ["08", "09"], "kai": ["05", "04"], "day_range": (1, 5), "R": "11"},  # 秋華賞: 京都5回
    "nhkmile":   {"places": ["05"], "kai": ["02"], "day_range": (4, 8), "R": "11"},  # NHKマイル: 東京2回
    "asahi_fs":  {"places": ["09", "06"], "kai": ["05", "04"], "day_range": (5, 9), "R": "11"},  # 朝日杯FS: 阪神5回
    "hopeful":   {"places": ["06"], "kai": ["05"], "day_range": (7, 10), "R": "11"},  # ホープフルS: 中山5回
}

# 該当馬種・開催年(=生年+offset)
# race -> 生年からの開催年オフセット, 性別
RACE_META = {
    "satsuki":  {"offset": 3, "sex": "牡"},
    "kikuka":   {"offset": 3, "sex": "牡"},
    "sakura":   {"offset": 3, "sex": "牝"},
    "syuka":    {"offset": 3, "sex": "牝"},
    "nhkmile":  {"offset": 3, "sex": None},
    "asahi_fs": {"offset": 2, "sex": None},  # 2歳暮
    "hopeful":  {"offset": 2, "sex": None},  # 2歳暮
}


def fetch_race_info(rid):
    """指定race_idのレース情報を取得。レース名と1着horse_idを返す。"""
    try:
        soup = _get_soup(f"https://db.netkeiba.com/race/{rid}/")
        title = soup.find("title")
        title_text = title.text if title else ""
        tbl = soup.find("table", class_="race_table_01")
        if not tbl:
            return None
        rows = tbl.find_all("tr")
        if len(rows) < 2:
            return None
        # 1着の馬名
        cols = rows[1].find_all("td")
        if len(cols) < 4:
            return None
        winner_name = cols[3].get_text(strip=True)
        return {"title": title_text, "winner": winner_name}
    except Exception:
        return None


def find_race_id(race_name, year):
    """指定レースの該当年のrace_idを試行錯誤で探す。"""
    pat = RACE_PATTERNS[race_name]
    for place in pat["places"]:
        for kai in pat["kai"]:
            for day in range(pat["day_range"][0], pat["day_range"][1] + 1):
                rid = f"{year}{place}{kai}{day:02d}{pat['R']}"
                info = fetch_race_info(rid)
                if info and info["title"]:
                    # レース名一致確認
                    t = info["title"]
                    # 各重賞のキーワードでフィルタ
                    keywords = {
                        "satsuki": "皐月",
                        "kikuka": "菊花",
                        "sakura": "桜花",
                        "syuka": "秋華",
                        "nhkmile": "NHK",
                        "asahi_fs": "朝日杯",
                        "hopeful": "ホープフル",
                    }
                    if keywords[race_name] in t:
                        return rid, t.split("|")[0].strip()
                time.sleep(0.3)
    return None, None


def fetch_top5(rid):
    """race_idから1-5着のhorse_idを取得。"""
    try:
        soup = _get_soup(f"https://db.netkeiba.com/race/{rid}/")
        tbl = soup.find("table", class_="race_table_01")
        if not tbl:
            return []
        ids = []
        for r in tbl.find_all("tr")[1:6]:
            for a in r.find_all("a"):
                href = a.get("href", "")
                if "/horse/" in href:
                    hid = href.strip("/").split("/")[-1]
                    if hid.isdigit():
                        ids.append(hid)
                        break
        return ids
    except Exception:
        return []


def main():
    # 既存キャッシュ読み込み
    rid_cache = json.load(open(CACHE_RID, encoding="utf-8")) if os.path.exists(CACHE_RID) else {}
    top5_cache = json.load(open(CACHE_TOP5, encoding="utf-8")) if os.path.exists(CACHE_TOP5) else {}

    # 生年 2015-2023
    for birth in range(2015, 2024):
        for race in RACE_META:
            meta = RACE_META[race]
            race_year = birth + meta["offset"]
            key = f"{race}_{birth}"
            if key in rid_cache and rid_cache[key]:
                rid = rid_cache[key]
                print(f"[CACHE] {key} -> {rid}")
            else:
                print(f"[SEARCH] {key} (year={race_year})...", end=" ")
                rid, title = find_race_id(race, race_year)
                if rid:
                    print(f"FOUND {rid} ({title})")
                    rid_cache[key] = rid
                    json.dump(rid_cache, open(CACHE_RID, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                else:
                    print("NOT FOUND")
                    rid_cache[key] = None
                    json.dump(rid_cache, open(CACHE_RID, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    continue
            # TOP5取得
            if key not in top5_cache and rid:
                top5 = fetch_top5(rid)
                top5_cache[key] = top5
                json.dump(top5_cache, open(CACHE_TOP5, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                print(f"        TOP5: {top5}")
                time.sleep(0.5)


if __name__ == "__main__":
    main()
