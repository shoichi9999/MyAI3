"""
特徴量エンジニアリングモジュール。

POG予測に重要な特徴量を生成する（デビュー前に入手可能な情報のみ）:
- 血統スコア（父馬の産駒EI、母父馬の産駒EI、母馬の獲得賞金）
- 生まれ月（早生まれほど有利）
- 親年齢（父・母の産駒時年齢）
- 祖父母年齢（父父・父母・母父・母母の産駒時年齢）
- 曾祖父母年齢（3世代目、8頭分の産駒時年齢）
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


# 種牡馬リーディング（産駒賞金・EI）
SIRE_LEADING = _load_json("data/sire_leading_2024.json")
# 母父馬リーディング（産駒賞金・EI）
BMS_LEADING = _load_json("data/bms_leading_2024.json")
# 母馬の獲得賞金
DAM_PRIZES = _load_json("data/dam_prizes.json")
# 生年月日キャッシュ（世代別）
BIRTH_DATES_CACHE = {}
# 親年齢キャッシュ（世代別）
PARENT_AGES_CACHE = {}


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

# ヒューリスティックスコアの重み配分（血統特化）
WEIGHT_SIRE_EI = 0.40
WEIGHT_DAM_PRIZE = 0.40
WEIGHT_BMS_EI = 0.20
WEIGHT_TRAINER = 0.00
WEIGHT_BREEDER = 0.00


def get_sire_ei(sire_name: str) -> float:
    """種牡馬のEI（アーニングインデックス）を返す。"""
    if not sire_name:
        return 0.0
    data = SIRE_LEADING.get(sire_name, {})
    try:
        return float(data.get("ei", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_bms_ei(bms_name: str) -> float:
    """母父馬のEI（アーニングインデックス）を返す。"""
    if not bms_name:
        return 0.0
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


def get_parent_age(horse_id: str, birth_year: int, parent: str = "sire") -> float:
    """親の産駒時年齢を返す。取得できない場合はNone。"""
    pa_cache = _load_parent_ages(birth_year)
    data = pa_cache.get(str(horse_id), {})
    by = data.get(f"{parent}_birth_year")
    if by:
        return birth_year - by
    return None


def build_feature_matrix(horses_df: pd.DataFrame, birth_year: int = None) -> pd.DataFrame:
    """
    全馬の特徴量マトリクスを構築する。

    血統スコア（父EI、母父EI、母馬獲得賞金）、生まれ月、親年齢を
    使って予測に使える特徴量を生成する。
    """
    feature_rows = []

    # 母馬賞金の中央値（賞金0の馬への補完用）
    dam_prizes_list = [v for v in DAM_PRIZES.values() if v > 0]
    dam_median = np.median(dam_prizes_list) if dam_prizes_list else 0.0

    # birth_yearの推定（horse_idの先頭4桁 or DataFrameから）
    if birth_year is None:
        sample_id = str(horses_df.iloc[0].get("horse_id", ""))
        if sample_id[:4].isdigit():
            birth_year = int(sample_id[:4])
        else:
            birth_year = 2024

    for _, horse in horses_df.iterrows():
        hid = horse.get("horse_id", "")
        row = {"horse_id": hid}

        # 基本情報
        row["horse_name"] = horse.get("horse_name", "")
        row["sex"] = 1 if horse.get("sex") == "牡" else (0 if horse.get("sex") == "牝" else 0.5)

        # 産駒成績ベースの血統スコア
        row["sire_ei"] = get_sire_ei(horse.get("sire", ""))
        row["bms_ei"] = get_bms_ei(horse.get("sire_of_dam", ""))

        # 母馬の獲得賞金（0の場合は中央値で補完）
        raw_dam_prize = get_dam_prize(horse.get("dam", ""))
        row["dam_prize"] = raw_dam_prize if raw_dam_prize > 0 else dam_median

        # 生まれ月（1-12、小さいほど有利）
        row["birth_month"] = get_birth_month(hid, birth_year)

        # 親年齢（父・母の産駒時年齢）
        sire_age = get_parent_age(hid, birth_year, "sire")
        dam_age = get_parent_age(hid, birth_year, "dam")
        row["sire_age"] = sire_age if sire_age is not None else 11.0  # 平均値で補完
        row["dam_age"] = dam_age if dam_age is not None else 10.5

        # 祖父母年齢（父父・父母・母父・母母の産駒時年齢）
        sire_sire_age = get_parent_age(hid, birth_year, "sire_sire")
        sire_dam_age = get_parent_age(hid, birth_year, "sire_dam")
        dam_sire_age = get_parent_age(hid, birth_year, "dam_sire")
        dam_dam_age = get_parent_age(hid, birth_year, "dam_dam")
        row["sire_sire_age"] = sire_sire_age if sire_sire_age is not None else 22.0
        row["sire_dam_age"] = sire_dam_age if sire_dam_age is not None else 21.0
        row["dam_sire_age"] = dam_sire_age if dam_sire_age is not None else 22.0
        row["dam_dam_age"] = dam_dam_age if dam_dam_age is not None else 21.0

        # 曾祖父母年齢（3世代目、8頭分の産駒時年齢）
        for ggp in [
            "sire_sire_sire", "sire_sire_dam",
            "sire_dam_sire", "sire_dam_dam",
            "dam_sire_sire", "dam_sire_dam",
            "dam_dam_sire", "dam_dam_dam",
        ]:
            age = get_parent_age(hid, birth_year, ggp)
            row[f"{ggp}_age"] = age if age is not None else 33.0

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)
