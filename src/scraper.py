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
RACE_BASE_URL = "https://race.netkeiba.com"

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


def fetch_horse_list_by_year(birth_year: int) -> pd.DataFrame:
    """
    指定した生年の馬一覧を取得する。
    netkeibaの種牡馬・繁殖牝馬ページや新馬戦結果から2歳馬を収集する。

    Parameters
    ----------
    birth_year : int
        生年（例: 2024）

    Returns
    -------
    pd.DataFrame
        馬ID、馬名を含むDataFrame
    """
    horses = []
    # 世代別馬一覧ページからスクレイピング
    # netkeibaでは /horse/list/ で世代検索が可能
    for page in range(1, 20):
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
            if len(cols) < 2:
                continue
            name_tag = cols[0].find("a")
            if not name_tag:
                continue
            href = name_tag.get("href", "")
            horse_id_match = re.search(r"/horse/(\w+)", href)
            if not horse_id_match:
                continue
            horse_id = horse_id_match.group(1)
            horse_name = name_tag.text.strip()
            horses.append({"horse_id": horse_id, "horse_name": horse_name})

    df = pd.DataFrame(horses)
    if not df.empty:
        df = df.drop_duplicates(subset=["horse_id"])
    return df


def fetch_horse_profile(horse_id: str) -> dict:
    """
    馬の詳細プロフィール情報を取得する。

    Parameters
    ----------
    horse_id : str
        netkeiba上の馬ID

    Returns
    -------
    dict
        馬のプロフィール情報
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
        rows = prof_table.find_all("tr")
        for row in rows:
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
                trainer_link = td.find("a")
                if trainer_link:
                    href = trainer_link.get("href", "")
                    tid = re.search(r"/trainer/(\w+)", href)
                    if tid:
                        profile["trainer_id"] = tid.group(1)
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
        # 通常、血統表の最初の2つのリンクが父・母
        blood_names = [a.text.strip() for a in links if a.text.strip()]
        if len(blood_names) >= 1:
            profile["sire"] = blood_names[0]  # 父
        if len(blood_names) >= 2:
            profile["dam"] = blood_names[1]  # 母
        if len(blood_names) >= 3:
            profile["sire_of_dam"] = blood_names[2]  # 母父

    # 性別・毛色
    p_tags = soup.find_all("p", class_="txt_01")
    for p in p_tags:
        text = p.text.strip()
        if "牡" in text or "牝" in text or "セン" in text:
            profile["sex"] = "牡" if "牡" in text else ("牝" if "牝" in text else "セン")
            break

    return profile


def fetch_horse_results(horse_id: str) -> pd.DataFrame:
    """
    馬の戦績（レース結果）を取得する。

    Parameters
    ----------
    horse_id : str
        netkeiba上の馬ID

    Returns
    -------
    pd.DataFrame
        レース結果のDataFrame
    """
    url = f"{BASE_URL}/horse/{horse_id}/"
    soup = _get_soup(url)

    results = []
    result_table = soup.find("table", class_="db_h_race_results")
    if not result_table:
        return pd.DataFrame()

    rows = result_table.find("tbody")
    if not rows:
        return pd.DataFrame()

    for row in rows.find_all("tr"):
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


def fetch_sire_stats(sire_name: str) -> dict:
    """
    種牡馬の産駒成績統計を取得する。

    Parameters
    ----------
    sire_name : str
        種牡馬名

    Returns
    -------
    dict
        種牡馬の統計情報
    """
    url = f"{BASE_URL}/?pid=horse_list&sire={sire_name}&sort=prize"
    try:
        soup = _get_soup(url)
    except Exception:
        return {"sire_name": sire_name}

    stats = {"sire_name": sire_name}

    # 産駒の賞金合計や勝率などの統計を計算
    table = soup.find("table", class_="nk_tb_common")
    if table:
        rows = table.find_all("tr")[1:]
        total_offspring = len(rows)
        stats["num_offspring_listed"] = total_offspring

    return stats


def fetch_trainer_stats(trainer_id: str) -> dict:
    """
    調教師の成績統計を取得する。

    Parameters
    ----------
    trainer_id : str
        調教師ID

    Returns
    -------
    dict
        調教師の統計情報
    """
    url = f"{BASE_URL}/trainer/{trainer_id}/"
    try:
        soup = _get_soup(url)
    except Exception:
        return {"trainer_id": trainer_id}

    stats = {"trainer_id": trainer_id}

    # 成績テーブルから勝率等を取得
    result_table = soup.find("table", class_="nk_tb_common")
    if result_table:
        rows = result_table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if cols:
                text = row.text.strip()
                if "勝率" in text or "連対率" in text:
                    stats["summary"] = text

    return stats


def scrape_all_2yo_data(birth_year: int, max_horses: Optional[int] = None) -> dict:
    """
    2歳馬のデータを一括取得するメイン関数。

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
    horse_list = fetch_horse_list_by_year(birth_year)

    if horse_list.empty:
        print("[WARN] 馬一覧の取得に失敗しました。")
        return {"horses": pd.DataFrame(), "results": pd.DataFrame()}

    if max_horses:
        horse_list = horse_list.head(max_horses)

    print(f"  取得対象: {len(horse_list)}頭")

    profiles = []
    all_results = []

    for i, row in horse_list.iterrows():
        hid = row["horse_id"]
        hname = row["horse_name"]
        print(f"  [{i+1}/{len(horse_list)}] {hname} ({hid}) の情報を取得中...")

        try:
            prof = fetch_horse_profile(hid)
            profiles.append(prof)
        except Exception as e:
            print(f"    [WARN] プロフィール取得失敗: {e}")

        try:
            res = fetch_horse_results(hid)
            if not res.empty:
                all_results.append(res)
        except Exception as e:
            print(f"    [WARN] 戦績取得失敗: {e}")

    profiles_df = pd.DataFrame(profiles)
    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # CSVに保存
    profiles_df.to_csv(f"data/horses_{birth_year}.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(f"data/results_{birth_year}.csv", index=False, encoding="utf-8-sig")

    print(f"=== 完了: {len(profiles_df)}頭のデータを取得 ===")
    return {"horses": profiles_df, "results": results_df}
