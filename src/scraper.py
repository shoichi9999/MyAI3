"""
netkeiba.comから産駒データをスクレイピングするモジュール。

対象データ:
- 馬名、性別、血統情報（父・母・母父）+ 各horse_id・生年
- 調教師、馬主、生産者、賞金
- プロフィール情報（生年月日、セリ価格）
- 産駒番号（母馬ページから何番仔かを取得）
"""

import json
import os
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# スレッドセーフなグローバルレートリミッター
_request_lock = threading.Lock()
_next_request_time = 0.0
_MIN_REQUEST_GAP = 0.5  # 全スレッド共通の最小間隔（秒）— 約2 req/s

DEFAULT_MAX_WORKERS = 5

# コネクションプーリング用セッション
_session = requests.Session()
_session.headers.update(HEADERS)


def _rate_limited_sleep():
    """全スレッド共通のリクエスト間隔を強制する（スレッドセーフ）。

    ロック内でスケジュール時刻を予約し、ロック外でsleepすることで
    複数スレッドが並行して待機できる。
    """
    global _next_request_time
    with _request_lock:
        now = time.monotonic()
        scheduled = max(now, _next_request_time)
        _next_request_time = scheduled + _MIN_REQUEST_GAP
    wait = scheduled - time.monotonic()
    if wait > 0:
        time.sleep(wait)


def concurrent_fetch(
    items: list,
    fetch_fn,
    max_workers: int = DEFAULT_MAX_WORKERS,
    label: str = "items",
    on_result=None,
    save_interval: int = 50,
    save_fn=None,
) -> dict:
    """
    アイテムを並列取得する汎用ヘルパー。

    Parameters
    ----------
    items : list
        取得対象のキーリスト
    fetch_fn : callable(item) -> result
        各アイテムの取得関数（レートリミッター呼び出し含む）
    max_workers : int
        並列ワーカー数（デフォルト3）
    label : str
        進捗表示用ラベル
    on_result : callable(item, result, completed_count) -> None
        結果を受け取るコールバック（Lock外から呼ばれるため、内部でLockを取得すること）
    save_interval : int
        何件ごとにsave_fnを呼ぶか
    save_fn : callable() -> None
        定期保存用コールバック

    Returns
    -------
    dict : {item: result}
    """
    results = {}
    completed = 0
    errors = 0
    lock = threading.Lock()

    def _worker(item):
        return item, fetch_fn(item)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, item): item for item in items}

        for future in as_completed(futures):
            item = futures[future]
            try:
                _, result = future.result()
                with lock:
                    results[item] = result
                    completed += 1
                    if on_result:
                        on_result(item, result, completed)
                    if save_fn and completed % save_interval == 0:
                        save_fn()
                        print(f"  {completed}/{len(items)} {label} 処理済み (エラー: {errors})")
            except Exception as e:
                with lock:
                    results[item] = None
                    completed += 1
                    errors += 1
                print(f"  [ERROR] {item}: {e}")

    if save_fn:
        save_fn()
    print(f"  完了: {completed}/{len(items)} {label} (エラー: {errors})")
    return results


def _get_soup(url: str, max_retries: int = 4) -> BeautifulSoup:
    """URLからBeautifulSoupオブジェクトを取得する。

    サーバーが断続的にHTTP 400/403を返すことがあるため、
    指数バックオフでリトライを行う。
    """
    global _session
    for attempt in range(max_retries):
        _rate_limited_sleep()
        try:
            resp = _session.get(url, timeout=30)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8秒
                print(f"  [WARN] 接続エラー (attempt {attempt+1}/{max_retries}): {e} — {wait}秒待機")
                time.sleep(wait)
                continue
            raise
        if resp.status_code == 200:
            resp.encoding = "EUC-JP"
            return BeautifulSoup(resp.text, "lxml")
        # HTTP 400/403等 — セッションを再生成して指数バックオフリトライ
        wait = 2 ** (attempt + 1)
        print(f"  [WARN] HTTP {resp.status_code} (attempt {attempt+1}/{max_retries}): {url[:80]}... — {wait}秒待機")
        if attempt < max_retries - 1:
            time.sleep(wait)
            _session = requests.Session()
            _session.headers.update(HEADERS)
    # 最終試行の結果を返す（パースエラーになるが呼び出し元で空テーブル扱い）
    print(f"  [ERROR] {max_retries}回リトライ後も失敗: HTTP {resp.status_code} — {url[:80]}")
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


def _fetch_horse_list_page(birth_year: int, page: int) -> list[dict]:
    """馬一覧の単一ページを取得してパースする。"""
    url = (
        f"{BASE_URL}/horse/list.html"
        f"?year={birth_year}"
        f"&sort=prize-desc&limit=100&page={page}"
        f"&range=all&state=all&match=p"
    )
    soup = _get_soup(url)

    table = soup.find("table", class_="nk_tb_common")
    if not table:
        return []

    rows = table.find_all("tr")[1:]
    if not rows:
        return []

    result = []
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

        sire_id = _extract_id_from_param(cols[6], "sire_id")
        dam_id = _extract_id_from_param(cols[7], "mare_id")
        bms_id = _extract_id_from_param(cols[8], "bms_id")

        result.append({
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

    return result


def fetch_horse_list_by_year(birth_year: int, max_pages: int = None) -> pd.DataFrame:
    """
    指定した生年の馬一覧を取得する。

    /horse/list.html エンドポイントを使用し、父・母・母父の
    horse_id も同時に取得する（日本産馬はIDの先頭4桁が生年）。
    海外馬はプロフィールページから生年を取得する。

    Parameters
    ----------
    birth_year : int
        生年（例: 2024）
    max_pages : int, optional
        最大取得ページ数。Noneの場合は全ページ取得。
    """
    # ページ1を取得してデータの存在を確認
    first_page = _fetch_horse_list_page(birth_year, 1)
    if not first_page:
        print("  データが見つかりませんでした")
        return pd.DataFrame()

    horses = list(first_page)
    print(f"  ページ 1: {len(first_page)}頭取得")

    # 残りのページを並列取得
    if len(first_page) >= 100 and (max_pages is None or max_pages > 1):
        end_page = max_pages if max_pages else 100
        remaining = list(range(2, end_page + 1))

        results = concurrent_fetch(
            items=remaining,
            fetch_fn=lambda p: _fetch_horse_list_page(birth_year, p),
            label="ページ",
        )

        for page_num in sorted(results.keys()):
            page_data = results[page_num]
            if page_data is None:
                continue  # エラーページはスキップ
            if len(page_data) == 0:
                break  # 空ページ到達 = データ終端
            horses.extend(page_data)

    print(f"  全{len(horses)}頭取得")

    df = pd.DataFrame(horses)
    if not df.empty:
        df = df.drop_duplicates(subset=["horse_id"])

        # JRA登録馬のみに絞り込み（horse_idが数字のみ）
        before = len(df)
        df = df[df["horse_id"].str.match(r"^\d+$")].reset_index(drop=True)
        excluded = before - len(df)
        if excluded:
            print(f"  JRA登録馬に絞り込み: {before}頭 → {len(df)}頭（{excluded}頭除外）")

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
            for col, id_col in [("sire_birth_year", "sire_id"),
                                ("dam_birth_year", "dam_id"),
                                ("bms_birth_year", "bms_id")]:
                mask = df[col].isna() & df[id_col].isin(resolved.keys())
                df.loc[mask, col] = df.loc[mask, id_col].map(resolved)

        na_sire = df["sire_birth_year"].isna().sum()
        na_dam = df["dam_birth_year"].isna().sum()
        na_bms = df["bms_birth_year"].isna().sum()
        if na_sire or na_dam or na_bms:
            print(f"  生年不明（残り）: 父={na_sire}, 母={na_dam}, 母父={na_bms}")

    return df


# ------------------------------------------------------------------
# 海外馬の生年解決
# ------------------------------------------------------------------

_FOREIGN_BY_CACHE_FILE = "data/foreign_birth_years.json"


def _load_foreign_by_cache() -> dict[str, int | None]:
    """ファイルから海外馬生年キャッシュを読み込む。"""
    if os.path.exists(_FOREIGN_BY_CACHE_FILE):
        with open(_FOREIGN_BY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_foreign_by_cache(cache: dict[str, int | None]):
    """海外馬生年キャッシュをファイルに保存する。"""
    with open(_FOREIGN_BY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _resolve_foreign_birth_years(horse_ids: set[str]) -> dict[str, int]:
    """海外馬のhorse_idセットからプロフィールページ経由で生年を一括解決する。"""
    cache = _load_foreign_by_cache()
    resolved = {}
    to_fetch = []

    for hid in horse_ids:
        if hid in cache:
            if cache[hid] is not None:
                resolved[hid] = cache[hid]
            continue
        if hid[:4].isdigit():
            resolved[hid] = int(hid[:4])
            continue
        to_fetch.append(hid)

    if not to_fetch:
        print(f"    全てキャッシュ済: {len(resolved)}/{len(horse_ids)}頭")
        return resolved

    cache_lock = threading.Lock()

    def on_result(hid, by, _idx):
        with cache_lock:
            cache[hid] = by
            if by is not None:
                resolved[hid] = by

    def save():
        with cache_lock:
            _save_foreign_by_cache(dict(cache))

    concurrent_fetch(
        items=to_fetch,
        fetch_fn=_fetch_birth_year_from_profile,
        label="海外馬",
        on_result=on_result,
        save_interval=50,
        save_fn=save,
    )

    skipped = len(horse_ids) - len(to_fetch)
    print(f"    解決: {len(resolved)}/{len(horse_ids)}頭（キャッシュ済: {skipped}, 新規取得: {len(to_fetch)}）")
    return resolved


def _fetch_birth_year_from_profile(horse_id: str) -> int | None:
    """外国産馬のプロフィールページから生年を取得する。"""
    try:
        _rate_limited_sleep()
        url = f"{BASE_URL}/horse/{horse_id}/"
        resp = _session.get(url, timeout=30)
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
                        return int(m.group(1))
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# プロフィール（生年月日・セリ価格）
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


def fetch_horse_profile(horse_id: str) -> dict:
    """
    プロフィールページから生年月日・セリ価格を取得する。

    Returns
    -------
    dict
        {"birth_date": str|None, "sale_price": float|None}
    """
    url = f"{BASE_URL}/horse/{horse_id}/"
    soup = _get_soup(url)

    result = {"birth_date": None, "sale_price": None}

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

    return result


# ------------------------------------------------------------------
# 産駒番号（母馬ページから何番仔かを取得）
# ------------------------------------------------------------------

def fetch_dam_foal_list(dam_id: str) -> list[str]:
    """
    繁殖牝馬ページ (/horse/mare/{id}/) から産駒のhorse_idリストを生年順で取得する。

    Returns
    -------
    list[str]
        産駒のhorse_idリスト（生年順）
    """
    url = f"{BASE_URL}/horse/mare/{dam_id}/"
    soup = _get_soup(url)

    foal_ids = []
    table = soup.find("table", class_="nk_tb_common")
    if table:
        for row in table.find_all("tr")[1:]:
            for a_tag in row.find_all("a"):
                href = a_tag.get("href", "")
                m = re.search(r"/horse/(\w+)/", href)
                if m and m.group(1) != dam_id:
                    foal_ids.append(m.group(1))
                    break

    return foal_ids
