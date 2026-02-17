"""
netkeiba.comから2歳馬の情報をスクレイピングするモジュール。

対象データ:
- 馬名、性別、生年月日
- 父馬・母馬・母父馬（血統情報）
- 調教師、馬主、生産者
- 戦績（着順、賞金、レース名、距離、馬場状態等）
- セリ取引価格
"""

import time
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd


BASE_URL = "https://db.netkeiba.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# リクエスト間隔（秒）- サーバー負荷軽減のため
REQUEST_INTERVAL = 1.5


def _get_soup(url: str) -> BeautifulSoup:
    """URLからBeautifulSoupオブジェクトを取得する。"""
    time.sleep(REQUEST_INTERVAL)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.encoding = "EUC-JP"
    return BeautifulSoup(resp.text, "lxml")


def fetch_horse_list_by_year(birth_year: int, max_pages: int = 20) -> pd.DataFrame:
    """
    指定した生年の馬一覧を取得する。
    一覧ページから馬名・性別・血統・調教師・賞金を直接取得する。

    Parameters
    ----------
    birth_year : int
        生年（例: 2024）
    max_pages : int
        最大取得ページ数

    Returns
    -------
    pd.DataFrame
        馬の基本情報を含むDataFrame
    """
    horses = []
    for page in range(1, max_pages + 1):
        url = (
            f"{BASE_URL}/?pid=horse_list"
            f"&birthyear={birth_year}"
            f"&sort=prize&list=100&page={page}"
        )
        try:
            soup = _get_soup(url)
        except Exception as e:
            print(f"[WARN] ページ {page} の取得に失敗: {e}")
            break

        table = soup.find("table", class_="nk_tb_common")
        if not table:
            break

        rows = table.find_all("tr")[1:]  # ヘッダーを除く
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 12:
                continue

            # [1]: 馬名 + horse_id
            name_tag = cols[1].find("a")
            if not name_tag:
                continue
            href = name_tag.get("href", "")
            horse_id_match = re.search(r"/horse/(\w+)", href)
            if not horse_id_match:
                continue

            horse_id = horse_id_match.group(1)
            horse_name = name_tag.text.strip()

            # [2]: 性別
            sex = cols[2].text.strip()

            # [5]: 調教師
            trainer_text = cols[5].text.strip()
            trainer_link = cols[5].find("a")
            trainer_id = ""
            if trainer_link:
                tid_match = re.search(r"/trainer/\w+/(\w+)", trainer_link.get("href", ""))
                if tid_match:
                    trainer_id = tid_match.group(1)

            # [6]: 父
            sire = cols[6].text.strip()

            # [7]: 母
            dam = cols[7].text.strip()

            # [8]: 母父
            sire_of_dam = cols[8].text.strip()

            # [6]: 父
            sire_age_link = cols[6].find("a")
            sire_age = ""
            if sire_age_link:
                tid_match = re.search(r"sire_id=(\d{4})", trainer_link.get("href", ""))
                if tid_match:
                    sire_age = birth_year - tid_match.group(1)

            # [7]: 母
            dam_age_link = cols[7].text.strip()
            dam_age = ""
            if dam_age_link:
                tid_match = re.search(r"mare_id=(\d{4})", trainer_link.get("href", ""))
                if tid_match:
                    dam_age = birth_year - tid_match.group(1)

            # [8]: 母父
            sire_of_dam_age_link = cols[8].text.strip()
            sire_of_dam_age = ""
            if sire_of_dam_age_link:
                tid_match = re.search(r"bms_id=(\d{4})", trainer_link.get("href", ""))
                if tid_match:
                    sire_of_dam_age = birth_year - tid_match.group(1)

            # [9]: 馬主
            owner = cols[9].text.strip()

            # [10]: 生産者
            breeder = cols[10].text.strip()

            # [11]: 総賞金
            total_prize = cols[11].text.strip()

            horses.append({
                "horse_id": horse_id,
                "horse_name": horse_name,
                "sex": sex,
                "trainer": trainer_text,
                "trainer_id": trainer_id,
                "sire": sire,
                "dam": dam,
                "sire_of_dam": sire_of_dam,
                "sire_age": sire_age,
                "dam_age": dam_age,
                "sire_of_dam_age": sire_of_dam_age,
                "owner": owner,
                "breeder": breeder,
                "total_prize": total_prize,
            })

        print(f"  ページ {page}: {len(rows)}頭取得")

    df = pd.DataFrame(horses)
    if not df.empty:
        df = df.drop_duplicates(subset=["horse_id"])
    return df


def fetch_horse_profile(horse_id: str) -> dict:
    """
    馬の詳細プロフィール情報を取得する（セリ価格等、一覧にない情報用）。
    """
    url = f"{BASE_URL}/horse/{horse_id}/"
    soup = _get_soup(url)

    profile = {"horse_id": horse_id}

    # 馬名
    name_tag = soup.find("div", class_="horse_title")
    if name_tag:
        h1 = name_tag.find("h1")
        if h1:
            profile["horse_name"] = h1.text.strip()

    # プロフィールテーブル
    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table:
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            key = th.text.strip()
            val = td.text.strip()

            if "生年月日" in key:
                profile["birth_date"] = val
            elif "調教師" in key:
                profile["trainer"] = val
            elif "馬主" in key:
                profile["owner"] = val
            elif "生産者" in key:
                profile["breeder"] = val
            elif "セリ取引価格" in key:
                profile["sale_price"] = val
            elif "獲得賞金" in key:
                profile["total_prize"] = val

    # 血統情報
    blood_table = soup.find("table", class_="blood_table")
    if blood_table:
        links = blood_table.find_all("a")
        blood_names = [a.text.strip() for a in links if a.text.strip()]
        if len(blood_names) >= 1:
            profile["sire"] = blood_names[0]
        if len(blood_names) >= 2:
            profile["dam"] = blood_names[1]
        if len(blood_names) >= 3:
            profile["sire_of_dam"] = blood_names[2]

    return profile


def fetch_horse_results(horse_id: str) -> pd.DataFrame:
    """馬の戦績（レース結果）を取得する。"""
    url = f"{BASE_URL}/horse/{horse_id}/"
    soup = _get_soup(url)

    results = []
    result_table = soup.find("table", class_="db_h_race_results")
    if not result_table:
        return pd.DataFrame()

    tbody = result_table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    for row in tbody.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 20:
            continue
        try:
            race_data = {
                "horse_id": horse_id,
                "date": cols[0].text.strip(),
                "venue": cols[1].text.strip(),
                "weather": cols[2].text.strip(),
                "race_number": cols[3].text.strip(),
                "race_name": cols[4].text.strip(),
                "num_horses": cols[6].text.strip(),
                "post_position": cols[7].text.strip(),
                "odds": cols[8].text.strip(),
                "popularity": cols[9].text.strip(),
                "finish_position": cols[10].text.strip(),
                "jockey": cols[11].text.strip(),
                "weight_carried": cols[12].text.strip(),
                "distance": cols[13].text.strip(),
                "track_condition": cols[14].text.strip(),
                "time": cols[17].text.strip(),
                "margin": cols[18].text.strip(),
                "horse_weight": cols[22].text.strip() if len(cols) > 22 else "",
                "prize": cols[27].text.strip() if len(cols) > 27 else "0",
            }
            results.append(race_data)
        except (IndexError, AttributeError):
            continue

    return pd.DataFrame(results)


def scrape_all_2yo_data(birth_year: int, max_horses: Optional[int] = None) -> dict:
    """
    2歳馬のデータを一括取得するメイン関数。
    一覧ページから効率的にデータを取得する。

    Parameters
    ----------
    birth_year : int
        対象の生年
    max_horses : int, optional
        取得する最大馬数（テスト用）

    Returns
    -------
    dict
        "horses": プロフィール一覧のDataFrame,
        "results": 全戦績のDataFrame
    """
    print(f"=== {birth_year}年生まれの馬一覧を取得中 ===")

    # 一覧ページから基本情報を一括取得
    max_pages = 20 if not max_horses else (max_horses // 100) + 1
    horse_list = fetch_horse_list_by_year(birth_year, max_pages=max_pages)

    if horse_list.empty:
        print("[WARN] 馬一覧の取得に失敗しました。")
        return {"horses": pd.DataFrame(), "results": pd.DataFrame()}

    if max_horses:
        horse_list = horse_list.head(max_horses)

    print(f"  取得完了: {len(horse_list)}頭")

    # CSVに保存
    horse_list.to_csv(f"data/horses_{birth_year}.csv", index=False, encoding="utf-8-sig")

    # 戦績はまだレース未出走の馬が多い場合は空
    results_df = pd.DataFrame()
    results_df.to_csv(f"data/results_{birth_year}.csv", index=False, encoding="utf-8-sig")

    print(f"=== 完了: {len(horse_list)}頭のデータを保存 ===")
    return {"horses": horse_list, "results": results_df}
