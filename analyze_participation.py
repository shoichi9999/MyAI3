"""
2015-2022年産（クラシック2018-2025）について、
予測TOP10/30/100の中で実際にダービー/オークスに出走した馬数を集計する。
"""
import sys, os, json, time
sys.path.insert(0, ".")
import pandas as pd
from src.scraper import _get_soup
from src.features import build_feature_matrix
from src.model import heuristic_score

CACHE = "data/race_starters_cache.json"

def fetch_starters(race_id: str) -> list[str]:
    url = f"https://db.netkeiba.com/race/{race_id}/"
    soup = _get_soup(url)
    table = soup.find("table", class_="race_table_01") or soup.find("table", class_="nk_tb_common")
    if not table:
        return []
    ids = []
    for r in table.find_all("tr")[1:]:
        for a in r.find_all("a"):
            href = a.get("href", "")
            if "/horse/" in href:
                hid = href.strip("/").split("/")[-1]
                if hid.isdigit():
                    ids.append(hid)
                    break
    return ids

def main():
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    rows = []
    print(f"{'生年':>4} {'レース':<6} {'starters':>8} {'TOP5':>4} {'TOP10':>5} {'TOP30':>5} {'TOP100':>6} {'TOP500':>6} {'avg_rank':>8}")
    print("-" * 70)
    # ダービー2018だけ末尾10R開催の特例
    DERBY_OVERRIDES = {2018: "201805021210"}
    for birth in range(2015, 2023):
        race_year = birth + 3
        for race_type in ["derby", "oaks"]:
            if race_type == "derby":
                rid = DERBY_OVERRIDES.get(race_year, f"{race_year}05021211")
            else:
                rid = f"{race_year}05021011"
            # 取得
            if rid not in cache:
                starters = fetch_starters(rid)
                cache[rid] = starters
                json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                time.sleep(1)
            else:
                starters = cache[rid]
            if not starters:
                print(f"{birth}  {race_type}: NO STARTERS (race_id={rid})")
                continue
            # 予測スコア計算
            horses = pd.read_csv(f"data/horses_{birth}.csv")
            sex = "牡" if race_type == "derby" else "牝"
            feats = build_feature_matrix(horses, birth_year=birth, sex_filter=sex)
            feats["score"] = heuristic_score(feats, race_type=race_type)
            feats = feats.sort_values("score", ascending=False).reset_index(drop=True)
            feats["rank"] = feats.index + 1
            id_to_rank = dict(zip(feats["horse_id"].astype(str), feats["rank"]))
            ranks = [id_to_rank[s] for s in starters if s in id_to_rank]
            n = len(starters)
            t5 = sum(1 for r in ranks if r <= 5)
            t10 = sum(1 for r in ranks if r <= 10)
            t30 = sum(1 for r in ranks if r <= 30)
            t100 = sum(1 for r in ranks if r <= 100)
            t500 = sum(1 for r in ranks if r <= 500)
            avg = sum(ranks) / len(ranks) if ranks else 0
            print(f"{birth}  {race_type:<6} {n:>8} {t5:>4} {t10:>5} {t30:>5} {t100:>6} {t500:>6} {avg:>8.0f}")
            rows.append({
                "birth": birth, "race": race_type, "n_starters": n,
                "ranked": len(ranks),
                "in_top5": t5, "in_top10": t10, "in_top30": t30, "in_top100": t100, "in_top500": t500,
                "avg_rank": round(avg, 1),
            })
    # サマリ
    print("\n=== Summary by race ===")
    df = pd.DataFrame(rows)
    for rt in ["derby", "oaks"]:
        sub = df[df["race"] == rt]
        if sub.empty: continue
        print(f"\n{rt}: 8 races")
        print(f"  Avg starters per race: {sub['n_starters'].mean():.1f}")
        print(f"  Avg matched (in our dataset): {sub['ranked'].mean():.1f}")
        print(f"  Avg # in TOP5:   {sub['in_top5'].mean():.2f}  / {sub['n_starters'].mean():.0f} starters")
        print(f"  Avg # in TOP10:  {sub['in_top10'].mean():.2f}  / {sub['n_starters'].mean():.0f} starters")
        print(f"  Avg # in TOP30:  {sub['in_top30'].mean():.2f}")
        print(f"  Avg # in TOP100: {sub['in_top100'].mean():.2f}")
        print(f"  Avg # in TOP500: {sub['in_top500'].mean():.2f}")

if __name__ == "__main__":
    main()
