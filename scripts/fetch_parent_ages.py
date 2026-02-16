"""指定世代の全馬の父母〜曾祖父母の生年を血統ページから取得してJSONに保存する。"""

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


# 外国産馬の生年キャッシュ（個別ページから取得済み分）
_foreign_birth_year_cache = {}


def _extract_birth_year(horse_id: str) -> int | None:
    """horse_idから生年を抽出する。日本産馬はID先頭4桁、外国産馬はプロフィールページから取得。"""
    if horse_id[:4].isdigit():
        return int(horse_id[:4])

    # 外国産馬（000a...形式）: キャッシュ確認
    if horse_id in _foreign_birth_year_cache:
        return _foreign_birth_year_cache[horse_id]

    # プロフィールページから生年月日を取得
    try:
        time.sleep(0.5)
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
                    m = re.search(r"(\d{4})年", td.text)
                    if m:
                        by = int(m.group(1))
                        _foreign_birth_year_cache[horse_id] = by
                        return by
    except Exception:
        pass

    _foreign_birth_year_cache[horse_id] = None
    return None


def fetch_pedigree_birth_years(horse_id: str) -> dict:
    """
    血統ページから3世代分の先祖の生年を取得する。

    血統テーブル構造:
      rowspan=16: 父(b_ml), 母(b_fml)                         ... 2頭
      rowspan=8:  父父, 母父(b_ml), 父母, 母母(b_fml)          ... 4頭
      rowspan=4:  曾祖父母(b_ml x4, b_fml x4)                 ... 8頭
    """
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "lxml")

    result = {
        # 親（2頭）
        "sire_birth_year": None,
        "dam_birth_year": None,
        # 祖父母（4頭）
        "sire_sire_birth_year": None,
        "sire_dam_birth_year": None,
        "dam_sire_birth_year": None,
        "dam_dam_birth_year": None,
        # 曾祖父母（8頭）
        "sire_sire_sire_birth_year": None,
        "sire_sire_dam_birth_year": None,
        "sire_dam_sire_birth_year": None,
        "sire_dam_dam_birth_year": None,
        "dam_sire_sire_birth_year": None,
        "dam_sire_dam_birth_year": None,
        "dam_dam_sire_birth_year": None,
        "dam_dam_dam_birth_year": None,
    }

    table = soup.find("table", class_="blood_table")
    if not table:
        return result

    # 各rowspanごとに出現順で収集
    gen1_ml, gen1_fml = [], []   # rowspan=16 (親)
    gen2_ml, gen2_fml = [], []   # rowspan=8  (祖父母)
    gen3_ml, gen3_fml = [], []   # rowspan=4  (曾祖父母)

    for td in table.find_all("td"):
        rs = td.get("rowspan")
        if not rs:
            continue
        rs = int(rs)
        if rs not in (16, 8, 4):
            continue

        cls = td.get("class", [])
        a = td.find("a")
        if not a:
            continue
        href = a.get("href", "")
        hid_match = re.search(r"/horse/(\w+)", href)
        if not hid_match:
            continue

        pid = hid_match.group(1)
        by = _extract_birth_year(pid)

        is_male = "b_ml" in cls
        is_female = "b_fml" in cls

        if rs == 16:
            (gen1_ml if is_male else gen1_fml).append(by)
        elif rs == 8:
            (gen2_ml if is_male else gen2_fml).append(by)
        elif rs == 4:
            (gen3_ml if is_male else gen3_fml).append(by)

    # 親
    if gen1_ml:
        result["sire_birth_year"] = gen1_ml[0]
    if gen1_fml:
        result["dam_birth_year"] = gen1_fml[0]

    # 祖父母 (出現順: 父父→母父(ml), 父母→母母(fml))
    if len(gen2_ml) >= 1:
        result["sire_sire_birth_year"] = gen2_ml[0]
    if len(gen2_ml) >= 2:
        result["dam_sire_birth_year"] = gen2_ml[1]
    if len(gen2_fml) >= 1:
        result["sire_dam_birth_year"] = gen2_fml[0]
    if len(gen2_fml) >= 2:
        result["dam_dam_birth_year"] = gen2_fml[1]

    # 曾祖父母 (出現順: 父父父→父母父→母父父→母母父(ml), 父父母→父母母→母父母→母母母(fml))
    ggp_ml_keys = [
        "sire_sire_sire_birth_year",
        "sire_dam_sire_birth_year",
        "dam_sire_sire_birth_year",
        "dam_dam_sire_birth_year",
    ]
    ggp_fml_keys = [
        "sire_sire_dam_birth_year",
        "sire_dam_dam_birth_year",
        "dam_sire_dam_birth_year",
        "dam_dam_dam_birth_year",
    ]
    for i, key in enumerate(ggp_ml_keys):
        if i < len(gen3_ml):
            result[key] = gen3_ml[i]
    for i, key in enumerate(ggp_fml_keys):
        if i < len(gen3_fml):
            result[key] = gen3_fml[i]

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_parent_ages.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])
    csv_path = f"data/horses_{year}.csv"
    cache_path = f"data/parent_ages_{year}.json"

    horses = pd.read_csv(csv_path)
    print(f"{year}年世代: {len(horses)}頭")

    # 既存キャッシュを読み込み（旧形式のキャッシュは無視して再取得）
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        # 曾祖父母データがあるキャッシュのみ有効
        for k, v in loaded.items():
            if "sire_sire_sire_birth_year" in v:
                cache[k] = v
        print(f"キャッシュ: {len(cache)}頭分あり (3世代データ)")

    total = len(horses)
    errors = 0

    for i, (_, row) in enumerate(horses.iterrows()):
        hid = str(row["horse_id"])
        if hid in cache:
            continue

        try:
            time.sleep(REQUEST_INTERVAL)
            result = fetch_pedigree_birth_years(hid)
            cache[hid] = result
        except Exception as e:
            cache[hid] = {k: None for k in [
                "sire_birth_year", "dam_birth_year",
                "sire_sire_birth_year", "sire_dam_birth_year",
                "dam_sire_birth_year", "dam_dam_birth_year",
                "sire_sire_sire_birth_year", "sire_sire_dam_birth_year",
                "sire_dam_sire_birth_year", "sire_dam_dam_birth_year",
                "dam_sire_sire_birth_year", "dam_sire_dam_birth_year",
                "dam_dam_sire_birth_year", "dam_dam_dam_birth_year",
            ]}
            errors += 1
            print(f"  [ERROR] {row['horse_name']}: {e}")

        if (i + 1) % 50 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            done = sum(1 for v in cache.values() if v.get("sire_birth_year"))
            print(f"  {i+1}/{total} 処理済み (取得成功: {done}, エラー: {errors})")

    # 最終保存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    done = sum(1 for v in cache.values() if v.get("sire_birth_year"))
    print(f"\n完了: {done}/{total}頭の親・祖父母・曾祖父母の年齢を取得 (エラー: {errors})")


if __name__ == "__main__":
    main()
