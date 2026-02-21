"""
特徴量エンジニアリングモジュール。

POG予測に重要な特徴量を生成する（デビュー前に入手可能な情報のみ）:
- 血統スコア（父馬の産駒EI、母父馬の産駒EI、母馬の獲得賞金）
- 生まれ月（早生まれほど有利）
- 親年齢（父・母の産駒時年齢）
- 調教師・馬主・牧場スコア（実績ベース）
- セリ価格・産駒番号
"""

import json
import os
import re

import numpy as np
import pandas as pd


# === リーディングデータ（産駒成績ベース） ===

def _load_json(path: str) -> dict:
    """JSONファイルを読み込む。存在しなければ空辞書を返す。"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 年度別リーディングキャッシュ
_LEADING_CACHE: dict[tuple[str, int], dict] = {}

# デフォルト（最新）のリーディングデータ — 予測時に使用
_DEFAULT_LEADING_YEAR = 2024
SIRE_LEADING = _load_json("data/sire_leading_2024.json")
BMS_LEADING = _load_json("data/bms_leading_2024.json")


def _load_leading(kind: str, year: int) -> dict:
    """年度別リーディングデータを読み込む（キャッシュ付き）。

    Parameters
    ----------
    kind : str
        "sire_leading" or "bms_leading"
    year : int
        リーディングの年度

    Returns
    -------
    dict
        {馬名: {rank, progeny_prize, ei, ...}}
    """
    key = (kind, year)
    if key not in _LEADING_CACHE:
        _LEADING_CACHE[key] = _load_json(f"data/{kind}_{year}.json")
    return _LEADING_CACHE[key]


def get_leading_year(birth_year: int) -> int:
    """POGドラフト時点で利用可能なリーディング年度を返す。

    生年Yの馬 → デビューはY+2年 → POGドラフトはY+2年春
    → 最新の通年データは Y+1 年。
    ただし該当ファイルがなければ最も近い年に降格。
    """
    target = birth_year + 1
    # 対象年のファイルが存在するかチェック
    for y in [target, target - 1, target + 1, _DEFAULT_LEADING_YEAR]:
        if os.path.exists(f"data/sire_leading_{y}.json"):
            return y
    return _DEFAULT_LEADING_YEAR
# 母馬の獲得賞金
DAM_PRIZES = _load_json("data/dam_prizes.json")
# 母馬産駒リスト（dam_id → [horse_id, ...]、生年順）
DAM_FOALS = _load_json("data/dam_foals.json")
# 生年月日キャッシュ（世代別）
BIRTH_DATES_CACHE = {}
# 親年齢キャッシュ（世代別）
PARENT_AGES_CACHE = {}
# 追加特徴量キャッシュ（セリ価格・産駒番号、世代別）
EXTRA_FEATURES_CACHE = {}


def _load_birth_dates(birth_year: int) -> dict:
    """生年月日キャッシュを読み込む。"""
    if birth_year not in BIRTH_DATES_CACHE:
        BIRTH_DATES_CACHE[birth_year] = _load_json(f"data/birth_dates_{birth_year}.json")
    return BIRTH_DATES_CACHE[birth_year]


def _load_parent_ages(birth_year: int) -> dict:
    """親年齢キャッシュを読み込む。"""
    if birth_year not in PARENT_AGES_CACHE:
        PARENT_AGES_CACHE[birth_year] = _load_json(f"data/parent_ages_{birth_year}.json")
    return PARENT_AGES_CACHE[birth_year]


def _load_extra_features(birth_year: int) -> dict:
    """追加特徴量キャッシュ（セリ価格・産駒番号）を読み込む。"""
    if birth_year not in EXTRA_FEATURES_CACHE:
        EXTRA_FEATURES_CACHE[birth_year] = _load_json(f"data/extra_features_{birth_year}.json")
    return EXTRA_FEATURES_CACHE[birth_year]


# 有力調教師スコア（2歳戦〜クラシック実績ベース）
ELITE_TRAINERS = {
    "国枝栄": 92,
    "堀宣行": 90,
    "藤沢和雄": 88,
    "友道康夫": 90,
    "中内田充正": 89,
    "木村哲也": 87,
    "手塚貴久": 85,
    "矢作芳人": 88,
    "池江泰寿": 86,
    "須貝尚介": 84,
    "萩原清": 83,
    "田中博康": 80,
    "高野友和": 82,
    "安田隆行": 83,
    "音無秀孝": 81,
    "斉藤崇史": 84,
    "武幸四郎": 82,
    "杉山晴紀": 81,
    "福永祐一": 85,
    "宮田敬介": 82,
    "吉岡辰弥": 78,
    "蛯名正義": 80,
    "四位洋文": 78,
}


# 有力馬主スコア（クラシック・重賞実績ベース）
ELITE_OWNERS = {
    "サンデーレーシング": 90,
    "金子真人ホールディングス": 88,
    "社台レースホース": 85,
    "シルクレーシング": 82,
    "キャロットファーム": 82,
    "ダノックス": 78,
    "藤田晋": 78,
    "前田晋二": 80,
    "近藤利一": 78,
    "近藤旬子": 78,
    "近藤英子": 75,
    "吉田勝己": 85,
    "吉田千津": 80,
    "吉田照哉": 80,
    "大塚亮一": 75,
    "石川達絵": 75,
    "前田幸治": 75,
    "前田葉子": 75,
    "三木正浩": 75,
    "窪田芳郎": 74,
    "キャピタル・システム": 74,
    "Ｇリビエール・レーシング": 74,
    "ＫＲジャパン": 76,
    "国本哲秀": 73,
    "岡田牧雄": 73,
    "寺田千代乃": 73,
    "西川光一": 75,
    "八木良司": 74,
    "土井肇": 76,
    "タマモ": 73,
    "ＴＯＲＡＣＩＮＧ": 78,
}


# 生産牧場スコア（POG上位輩出実績ベース）
ELITE_BREEDERS = {
    "ノーザンファーム": 95,
    "社台ファーム": 88,
    "社台コーポレーション白老ファーム": 85,
    "追分ファーム": 82,
    "ノースヒルズ": 80,
    "下河辺牧場": 78,
    "ダーレー・ジャパン・ファーム": 80,
    "レイクヴィラファーム": 75,
    "ケイアイファーム": 74,
    "岡田スタッド": 74,
    "ビッグレッドファーム": 73,
    "コスモヴューファーム": 72,
}

# ヒューリスティックスコアの重み配分（グリッドサーチ最適化済み）
WEIGHT_SIRE_EI = 0.225
WEIGHT_DAM_PRIZE = 0.075
WEIGHT_BMS_EI = 0.25


def get_sire_ei(sire_name: str, leading_year: int = None) -> float:
    """種牡馬のEI（アーニングインデックス）を返す。

    Parameters
    ----------
    sire_name : str
        種牡馬名
    leading_year : int, optional
        参照するリーディング年度。Noneならデフォルト（最新）を使用。
    """
    if not sire_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("sire_leading", leading_year).get(sire_name, {})
    else:
        data = SIRE_LEADING.get(sire_name, {})
    try:
        return float(data.get("ei", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_bms_ei(bms_name: str, leading_year: int = None) -> float:
    """母父馬のEI（アーニングインデックス）を返す。

    Parameters
    ----------
    bms_name : str
        母父馬名
    leading_year : int, optional
        参照するリーディング年度。Noneならデフォルト（最新）を使用。
    """
    if not bms_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("bms_leading", leading_year).get(bms_name, {})
    else:
        data = BMS_LEADING.get(bms_name, {})
    try:
        return float(data.get("ei", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_dam_prize(dam_name: str) -> float:
    """母馬の獲得賞金（万円）を返す。"""
    if not dam_name:
        return 0.0
    return DAM_PRIZES.get(dam_name, 0.0)


def calc_trainer_score(trainer_name: str) -> float:
    """調教師スコアを返す。"""
    if not trainer_name:
        return 50.0
    for key, score in ELITE_TRAINERS.items():
        if key in str(trainer_name):
            return score
    return 50.0


def calc_owner_score(owner_name: str) -> float:
    """馬主スコアを返す。"""
    if not owner_name:
        return 50.0
    for key, score in ELITE_OWNERS.items():
        if key in str(owner_name):
            return score
    return 50.0


def calc_breeder_score(breeder_name: str) -> float:
    """生産牧場スコアを返す。"""
    if not breeder_name:
        return 50.0
    for key, score in ELITE_BREEDERS.items():
        if key in str(breeder_name):
            return score
    return 50.0


def get_birth_month(horse_id: str, birth_year: int) -> int:
    """馬の生まれ月を返す（1-12）。取得できない場合は3（中央値）。"""
    bd_cache = _load_birth_dates(birth_year)
    bd = bd_cache.get(str(horse_id), "")
    if bd:
        m = re.search(r"(\d+)月", bd)
        if m:
            return int(m.group(1))
    return 3  # デフォルト: 3月


def _birth_year_from_id(horse_id: str) -> int | None:
    """horse_idの先頭4桁から生年を返す（日本産馬のみ）。"""
    if horse_id and str(horse_id)[:4].isdigit():
        return int(str(horse_id)[:4])
    return None


# 親の (CSVカラム名, JSONキャッシュのbirth_yearキー, JSONキャッシュのidキー)
_PARENT_MAP = {
    "sire": ("sire_birth_year", "sire_birth_year", "sire_id"),
    "dam":  ("dam_birth_year",  "dam_birth_year",  "dam_id"),
    "bms":  ("bms_birth_year",  "dam_sire_birth_year", "bms_id"),
}


def get_parent_age(horse_row, birth_year: int, parent: str = "sire") -> float | None:
    """親の産駒時年齢を返す。取得できない場合はNone。

    優先順:
      1. CSVカラム (sire_birth_year 等) から直接計算
      2. JSONキャッシュ (parent_ages_{year}.json) からフォールバック
    """
    csv_col, cache_by_key, cache_id_key = _PARENT_MAP.get(
        parent, (f"{parent}_birth_year", f"{parent}_birth_year", None)
    )

    # 1. CSVカラムから（DataFrameの行に含まれている場合）
    parent_by = horse_row.get(csv_col)
    if pd.notna(parent_by) and parent_by:
        try:
            return birth_year - int(parent_by)
        except (ValueError, TypeError):
            pass

    # 2. JSONキャッシュからフォールバック
    hid = str(horse_row.get("horse_id", ""))
    pa_cache = _load_parent_ages(birth_year)
    data = pa_cache.get(hid, {})

    by = data.get(cache_by_key)
    if by:
        return birth_year - by

    if cache_id_key:
        parent_id = data.get(cache_id_key)
        if parent_id:
            parent_by = _birth_year_from_id(parent_id)
            if parent_by:
                return birth_year - parent_by

    return None


def build_feature_matrix(horses_df: pd.DataFrame, birth_year: int = None) -> pd.DataFrame:
    """
    全馬の特徴量マトリクスを構築する。

    血統スコア（父EI、母父EI、母馬獲得賞金）、生まれ月、親年齢を
    使って予測に使える特徴量を生成する。

    EIデータは birth_year に対応するリーディング年度
    （= birth_year + 1、POGドラフト時点で利用可能な最新データ）を参照する。
    """
    feature_rows = []

    # birth_yearの推定（horse_idの先頭4桁 or DataFrameから）
    if birth_year is None:
        sample_id = str(horses_df.iloc[0].get("horse_id", ""))
        if sample_id[:4].isdigit():
            birth_year = int(sample_id[:4])
        else:
            birth_year = 2024

    # リーディング年度の決定（データリーク防止）
    leading_year = get_leading_year(birth_year)

    for _, horse in horses_df.iterrows():
        hid = horse.get("horse_id", "")
        row = {"horse_id": hid}

        # 基本情報
        row["horse_name"] = horse.get("horse_name", "")
        row["sex"] = 1 if horse.get("sex") == "牡" else (0 if horse.get("sex") == "牝" else 0.5)

        # 産駒成績ベースの血統スコア（年度別リーディングを参照）
        row["sire_ei"] = get_sire_ei(horse.get("sire", ""), leading_year)
        row["bms_ei"] = get_bms_ei(horse.get("sire_of_dam", ""), leading_year)

        # 母馬の獲得賞金（不明なら0 — 不明は不利な情報として扱う）
        row["dam_prize"] = get_dam_prize(horse.get("dam", ""))

        # 調教師・馬主・生産者スコア
        row["trainer_score"] = calc_trainer_score(horse.get("trainer", ""))
        row["owner_score"] = calc_owner_score(horse.get("owner", ""))
        row["breeder_score"] = calc_breeder_score(horse.get("breeder", ""))

        # 生まれ月（1-12、小さいほど有利）
        row["birth_month"] = get_birth_month(hid, birth_year)

        # 親年齢（父・母・母父の産駒時年齢）— 欠損はNaN（デフォルト値補完しない）
        sire_age = get_parent_age(horse, birth_year, "sire")
        dam_age = get_parent_age(horse, birth_year, "dam")
        dam_sire_age = get_parent_age(horse, birth_year, "bms")
        row["sire_age"] = sire_age if sire_age is not None else np.nan
        row["dam_age"] = dam_age if dam_age is not None else np.nan
        row["dam_sire_age"] = dam_sire_age if dam_sire_age is not None else np.nan

        # --- ドメイン知識ベースのバイナリ特徴量 ---
        # 年齢がNaNの場合はバイナリもNaN（デフォルト値で偽装しない）
        row["early_born"] = 1 if row["birth_month"] <= 4 else 0
        row["sire_young"] = (1 if sire_age <= 13 else 0) if sire_age is not None else np.nan
        row["dam_young"] = (1 if dam_age <= 13 else 0) if dam_age is not None else np.nan
        row["both_parents_young"] = (
            (1 if sire_age <= 13 and dam_age <= 13 else 0)
            if sire_age is not None and dam_age is not None else np.nan
        )
        row["sire_first_crop"] = (1 if sire_age <= 7 else 0) if sire_age is not None else np.nan
        row["dam_bms_gap_small"] = (
            (1 if (dam_sire_age - dam_age) <= 15 else 0)
            if dam_sire_age is not None and dam_age is not None else np.nan
        )

        # --- 追加特徴量（セリ価格・産駒番号） ---
        extra = _load_extra_features(birth_year).get(str(hid), {})
        sale_price = extra.get("sale_price")
        row["sale_price_log"] = np.log1p(sale_price) if sale_price else 0.0
        foal_number = extra.get("foal_number")
        row["foal_number"] = foal_number if foal_number else 3.0  # デフォルト: 3番仔

        # --- 母馬の繁殖入り年齢（初仔生年 - 母馬生年 = 引退時期の近似） ---
        dam_breeding_age = np.nan
        dam_by = horse.get("dam_birth_year")
        dam_id = str(horse.get("dam_id", "")) if pd.notna(horse.get("dam_id")) else ""
        if dam_id and pd.notna(dam_by):
            foal_list = DAM_FOALS.get(dam_id, [])
            if foal_list:
                # 産駒リストは降順（新しい順）→ 末尾が初仔
                first_foal_by = _birth_year_from_id(foal_list[-1])
                if first_foal_by is not None:
                    dam_breeding_age = first_foal_by - int(dam_by)
        row["dam_breeding_age"] = dam_breeding_age

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)
