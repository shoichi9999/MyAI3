"""
netkeiba.comから産駒データをスクレイピングするモジュール。

対象データ:
- 馬名、性別、血統情報（父・母・母父）
- 調教師、馬主、生産者、賞金
- 親のhorse_id・生年（父・母・母父）
- プロフィール情報（生年月日、セリ価格、母馬ID）
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

def _extract_id_from_param(td_tag, param_name: str) -> str:
    """tdタグ内のリンクからリクエストパラメータのIDを抽出する。"""
    for a in td_tag.find_all("a"):
        href = a.get("href", "")
        m = re.search(rf"{param_name}=(\w+)", href)
        if m:
            return m.group(1)
    return ""


def _birth_year_from_horse_id(horse_id: str) -> int | None:
    """horse_idの先頭4桁から生年を抽出する（日本産馬のみ）。"""
    if horse_id and horse_id[:4].isdigit():
        return int(horse_id[:4])
    return None


def fetch_horse_list_by_year(birth_year: int, max_pages: int = None) -> pd.DataFrame:
    """
    指定した生年の馬一覧を取得する。

    /horse/list.html エンドポイントを使用し、父・母・母父の
    horse_id も同時に取得する（日本産馬はIDの先頭4桁が生年）。

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
            f"{BASE_URL}/horse/list.html"
            f"?year={birth_year}"
            f"&sort=prize-desc&limit=100&page={page}"
            f"&range=all&state=all&match=p"
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

            # 父・母・母父のhorse_idをリンクパラメータから抽出
            sire_id = _extract_id_from_param(cols[6], "sire_id")
            dam_id = _extract_id_from_param(cols[7], "mare_id")
            bms_id = _extract_id_from_param(cols[8], "bms_id")

            horses.append({
                "horse_id": horse_id_match.group(1),
                "horse_name": name_tag.text.strip(),
                "sex": cols[2].text.strip(),
                "trainer": _extract_cell_name(cols[5]),
                "trainer_id": _extract_trainer_id(cols[5]),
                "sire": _extract_cell_name(cols[6]),
                "sire_id": sire_id,
                "dam": _extract_cell_name(cols[7]),
                "dam_id": dam_id,
                "sire_of_dam": _extract_cell_name(cols[8]),
                "bms_id": bms_id,
                "owner": _extract_cell_name(cols[9]),
                "breeder": _extract_cell_name(cols[10]),
                "total_prize": cols[11].text.strip(),
            })

        print(f"  ページ {page}: {len(rows)}頭取得")

    df = pd.DataFrame(horses)
    if not df.empty:
        df = df.drop_duplicates(subset=["horse_id"])

        # 日本産馬のhorse_idから親の生年を算出
        for col, id_col in [("sire_birth_year", "sire_id"),
                            ("dam_birth_year", "dam_id"),
                            ("bms_birth_year", "bms_id")]:
            df[col] = df[id_col].apply(_birth_year_from_horse_id)

        # 海外馬（生年不明）のユニークIDを収集してプロフィールから生年取得
        foreign_ids = set()
        for id_col in ["sire_id", "dam_id", "bms_id"]:
            for hid in df.loc[df[id_col].str[:4].apply(
                    lambda x: not str(x).isdigit() if pd.notna(x) else True), id_col]:
                if hid:
                    foreign_ids.add(hid)

        if foreign_ids:
            print(f"  海外馬の生年を取得中: {len(foreign_ids)}頭...")
            resolved = _resolve_foreign_birth_years(foreign_ids)
            # 解決結果を反映
            for col, id_col in [("sire_birth_year", "sire_id"),
                                ("dam_birth_year", "dam_id"),
                                ("bms_birth_year", "bms_id")]:
                mask = df[col].isna() & df[id_col].isin(resolved.keys())
                df.loc[mask, col] = df.loc[mask, id_col].map(resolved)

        # 生年不明の件数を表示
        na_sire = df["sire_birth_year"].isna().sum()
        na_dam = df["dam_birth_year"].isna().sum()
        na_bms = df["bms_birth_year"].isna().sum()
        if na_sire or na_dam or na_bms:
            print(f"  生年不明（残り）: 父={na_sire}, 母={na_dam}, 母父={na_bms}")

    return df


def _extract_cell_name(td_tag) -> str:
    """tdタグから最初のリンクテキスト（名前）を取得する。"""
    a = td_tag.find("a")
    if a:
        text = a.text.strip()
        if text and not text.startswith("["):
            return text
    return td_tag.text.strip()


def _extract_trainer_id(td_tag) -> str:
    link = td_tag.find("a")
    if link:
        m = re.search(r"/trainer/\w+/(\w+)", link.get("href", ""))
        if m:
            return m.group(1)
    return ""


# ------------------------------------------------------------------
# 海外馬の生年解決
# ------------------------------------------------------------------

_foreign_birth_year_cache: dict[str, int | None] = {}


def _resolve_foreign_birth_years(horse_ids: set[str]) -> dict[str, int]:
    """海外馬のhorse_idセットからプロフィールページ経由で生年を一括解決する。"""
    resolved = {}
    for i, hid in enumerate(horse_ids):
        by = _extract_birth_year(hid)
        if by is not None:
            resolved[hid] = by
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(horse_ids)} 処理済み")
    print(f"    解決: {len(resolved)}/{len(horse_ids)}頭")
    return resolved


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


def fetch_horse_profile(horse_id: str) -> dict:
    """
    プロフィールページから生年月日・セリ価格・母馬IDを一括取得する。

    1ページアクセスで3つのキャッシュ（birth_dates, extra_features, 母馬ID）を
    すべて賄える統合関数。

    Returns
    -------
    dict
        {"birth_date": str|None, "sale_price": float|None, "dam_id": str|None}
    """
    url = f"{BASE_URL}/horse/{horse_id}/"
    soup = _get_soup(url)

    result = {"birth_date": None, "sale_price": None, "dam_id": None}

    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table:
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            label = th.text.strip()
            if "生年月日" in label:
                result["birth_date"] = td.text.strip()
            elif "セリ取引価格" in label:
                result["sale_price"] = _parse_sale_price(td.text)

    # 母馬ID: db_prof_tableの「近親馬」ではなく、血統のAjax取得ができないので
    # 別途 fetch_parent_ids で取得する設計とする

    return result


def fetch_parent_ids(horse_id: str) -> dict:
    """
    血統ページから父・母・母父のhorse_idを取得する。

    祖父母以降は不要なので、rowspan=16（父・母）と rowspan=8 の
    母父のみを取得する軽量版。

    Returns
    -------
    dict
        {"sire_id": str|None, "dam_id": str|None, "bms_id": str|None}
    """
    url = f"{BASE_URL}/horse/ped/{horse_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "lxml")

    result = {"sire_id": None, "dam_id": None, "bms_id": None}

    table = soup.find("table", class_="blood_table")
    if not table:
        return result

    gen1_ml, gen1_fml = [], []
    gen2_ml = []

    for td in table.find_all("td"):
        rs = td.get("rowspan")
        if not rs:
            continue
        rs = int(rs)
        if rs not in (16, 8):
            continue

        cls = td.get("class", [])
        a = td.find("a")
        if not a:
            continue
        href = a.get("href", "")
        hid_match = re.search(r"/horse/(\w+)", href)
        if not hid_match:
            continue

        hid = hid_match.group(1)
        is_male = "b_ml" in cls

        if rs == 16:
            (gen1_ml if is_male else gen1_fml).append(hid)
        elif rs == 8 and is_male:
            gen2_ml.append(hid)

    if gen1_ml:
        result["sire_id"] = gen1_ml[0]
    if gen1_fml:
        result["dam_id"] = gen1_fml[0]
    if len(gen2_ml) >= 2:
        result["bms_id"] = gen2_ml[1]  # 2番目の牡系=母父

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
