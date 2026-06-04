"""sire_prizes.json に未登録の sire の賞金を netkeibaから取得して追加。"""
import sys, json, time, re
sys.path.insert(0, ".")
import pandas as pd
from src.scraper import _get_soup


def fetch_prize(horse_id):
    soup = _get_soup(f"https://db.netkeiba.com/horse/{horse_id}/")
    prof = soup.find("table", class_="db_prof_table")
    if not prof:
        return None
    central = 0.0
    local = 0.0
    for row in prof.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            if "獲得賞金" in label:
                num = re.sub(r"[^\d.]", "", value.replace("億", "0000").replace("万円", ""))
                # 形式 "17億5,655" は handle 必要
                m = re.match(r"(\d+)億([\d,]+)", value)
                if m:
                    val = int(m.group(1)) * 10000 + float(m.group(2).replace(",", ""))
                else:
                    m2 = re.match(r"([\d,]+)", value)
                    val = float(m2.group(1).replace(",", "")) if m2 else 0
                if "中央" in label:
                    central = val
                elif "地方" in label:
                    local = val
    return central + local


def main():
    sp = json.load(open("data/sire_prizes.json", encoding="utf-8"))

    # ユニーク sire一覧
    sires = {}  # sire_name -> sire_id
    for y in range(2015, 2026):
        try:
            df = pd.read_csv(f"data/horses_{y}.csv")
            for _, r in df.iterrows():
                n, i = r.get("sire"), r.get("sire_id")
                if pd.notna(n) and pd.notna(i) and n not in sires:
                    sires[n] = str(i)
        except FileNotFoundError:
            pass

    missing = [(n, sires[n]) for n in sires if n not in sp]
    print(f"未登録: {len(missing)}頭")

    for i, (name, hid) in enumerate(missing, 1):
        prize = fetch_prize(hid)
        if prize is not None:
            sp[name] = prize
            print(f"  {i}/{len(missing)}  {name}: {prize}")
        time.sleep(1.0)

    with open("data/sire_prizes.json", "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2)
    print(f"完了: {len(sp)}頭登録")


if __name__ == "__main__":
    main()
