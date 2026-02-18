"""
netkeiba.comから産駒データをスクレイピングするモジュール。

対象データ:
- 馬名、性別、血統情報（父・母・母父）
- 調教師、馬主、生産者、賞金
- 3世代分の先祖の生年（父母〜曾祖父母）
"""

import time
import re

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


# ------------------------------------------------------------------
# 馬一覧取得
# ------------------------------------------------------------------

def fetch_horse_list_by_year(birth_year: int, max_pages: int = None) -> pd.DataFrame:
    """
    指定した生年の馬一覧を取得する。

    Parameters
    ----------
    birth_year : int
        生年（例: 2024）
    max_pages : int, optional
        最大取得ページ数。Noneの場合はデータがなくなるまで全ページ取得。
    """
    horses = []
    page = 0
    while True:
        page += 1
        if max_pages is not None and page > max_pages:
            break
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

        rows = table.find_all("tr")[1:]
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 12:
                continue

            name_tag = cols[1].find("a")
            if not name_tag:
                continue
            href = name_tag.get("href", "")
            horse_id_match = re.search(r"/horse/(\w+)", href)
            if not horse_id_match:
                continue

            horses.append({
                "horse_id": horse_id_match.group(1),
                "horse_name": name_tag.text.strip(),
                "sex": cols[2].text.strip(),
                "trainer": cols[5].text.strip(),
                "trainer_id": _extract_trainer_id(cols[5]),
                "sire": cols[6].text.strip(),
                "dam": cols[7].text.strip(),
                "sire_of_dam": cols[8].text.strip(),
                "owner": cols[9].text.strip(),
                "breeder": cols[10].text.strip(),
                "total_prize": cols[11].text.strip(),
            })

        print(f"  ページ {page}: {len(rows)}頭取得")

    df = pd.DataFrame(horses)
    if not df.empty:
        df = df.drop_duplicates(subset=["horse_id"])
    return df


def _extract_trainer_id(td_tag) -> str:
    link = td_tag.find("a")
    if link:
        m = re.search(r"/trainer/\w+/(\w+)", link.get("href", ""))
        if m:
            return m.group(1)
    return ""


# ------------------------------------------------------------------
# 血統（3世代分の先祖の生年）
# ------------------------------------------------------------------

_foreign_birth_year_cache: dict[str, int | None] = {}


def _extract_birth_year(horse_id: str) -> int | None:
    """horse_idから生年を抽出する。日本産馬はID先頭4桁、外国産馬はプロフィールページから取得。"""
    if horse_id[:4].isdigit():
        return int(horse_id[:4])

    if horse_id in _foreign_birth_year_cache:
        return _foreign_birth_year_cache[horse_id]

    try:
        time.sleep(0.5)
        url = f"{BASE_URL}/horse/{horse_id}/"
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
      rowspan=16: 父(b_ml), 母(b_fml)
      rowspan=8:  父父, 母父(b_ml), 父母, 母母(b_fml)
      rowspan=4:  曾祖父母(b_ml x4, b_fml x4)
    """
    url = f"{BASE_URL}/horse/ped/{horse_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "lxml")

    result = {
        "sire_birth_year": None,
        "dam_birth_year": None,
        "sire_sire_birth_year": None,
        "sire_dam_birth_year": None,
        "dam_sire_birth_year": None,
        "dam_dam_birth_year": None,
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

    gen1_ml, gen1_fml = [], []
    gen2_ml, gen2_fml = [], []
    gen3_ml, gen3_fml = [], []

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

        by = _extract_birth_year(hid_match.group(1))
        is_male = "b_ml" in cls

        if rs == 16:
            (gen1_ml if is_male else gen1_fml).append(by)
        elif rs == 8:
            (gen2_ml if is_male else gen2_fml).append(by)
        elif rs == 4:
            (gen3_ml if is_male else gen3_fml).append(by)

    if gen1_ml:
        result["sire_birth_year"] = gen1_ml[0]
    if gen1_fml:
        result["dam_birth_year"] = gen1_fml[0]

    if len(gen2_ml) >= 1:
        result["sire_sire_birth_year"] = gen2_ml[0]
    if len(gen2_ml) >= 2:
        result["dam_sire_birth_year"] = gen2_ml[1]
    if len(gen2_fml) >= 1:
        result["sire_dam_birth_year"] = gen2_fml[0]
    if len(gen2_fml) >= 2:
        result["dam_dam_birth_year"] = gen2_fml[1]

    ggp_ml_keys = [
        "sire_sire_sire_birth_year", "sire_dam_sire_birth_year",
        "dam_sire_sire_birth_year", "dam_dam_sire_birth_year",
    ]
    ggp_fml_keys = [
        "sire_sire_dam_birth_year", "sire_dam_dam_birth_year",
        "dam_sire_dam_birth_year", "dam_dam_dam_birth_year",
    ]
    for i, key in enumerate(ggp_ml_keys):
        if i < len(gen3_ml):
            result[key] = gen3_ml[i]
    for i, key in enumerate(ggp_fml_keys):
        if i < len(gen3_fml):
            result[key] = gen3_fml[i]

    return result


# ------------------------------------------------------------------
# セリ価格・母馬ID・産駒番号
# ------------------------------------------------------------------

def _parse_sale_price(text: str) -> float | None:
    """セリ価格テキストを万円単位の数値に変換する。"""
    if not text or text.strip() in ("", "-"):
        return None
    text = text.replace(",", "").replace("　", "").replace(" ", "")
    total = 0.0
    m_oku = re.search(r"(\d+)億", text)
    if m_oku:
        total += int(m_oku.group(1)) * 10000
    m_man = re.search(r"(\d+)万", text)
    if m_man:
        total += int(m_man.group(1))
    return total if total > 0 else None


def fetch_horse_extra(horse_id: str) -> dict:
    """
    プロフィールページからセリ取引価格と母馬IDを取得する。

    Returns
    -------
    dict
        {"sale_price": float|None (万円), "dam_id": str|None}
    """
    url = f"{BASE_URL}/horse/{horse_id}/"
    soup = _get_soup(url)

    result = {"sale_price": None, "dam_id": None}

    # セリ取引価格
    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table:
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "セリ取引価格" in th.text:
                result["sale_price"] = _parse_sale_price(td.text)

    # 母馬ID（血統テーブルの b_fml rowspan=4 が母）
    blood_table = soup.find("table", class_="blood_table")
    if blood_table:
        for td in blood_table.find_all("td", class_="b_fml"):
            rs = td.get("rowspan")
            if rs and int(rs) == 4:
                a = td.find("a")
                if a:
                    href = a.get("href", "")
                    m = re.search(r"/horse/(\w+)", href)
                    if m:
                        result["dam_id"] = m.group(1)
                break

    return result


def fetch_dam_foal_list(dam_id: str) -> list[str]:
    """
    母馬のページから産駒のhorse_idリストを生年順で取得する。

    Returns
    -------
    list[str]
        産駒のhorse_idリスト（生年順）
    """
    url = f"{BASE_URL}/horse/{dam_id}/"
    soup = _get_soup(url)

    foal_ids = []

    # 産駒テーブル: "産駒" を含むヘッダーの直後のテーブルを探す
    for tag in soup.find_all(["h2", "h3", "h4", "div"]):
        if "産駒" in tag.get_text():
            table = tag.find_next("table")
            if table:
                for row in table.find_all("tr")[1:]:
                    for a_tag in row.find_all("a"):
                        href = a_tag.get("href", "")
                        m = re.search(r"/horse/(\w+)/", href)
                        if m and m.group(1) != dam_id:
                            foal_ids.append(m.group(1))
                            break
                return foal_ids

    return foal_ids
