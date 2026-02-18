"""
特徴量エンジニアリングモジュール。

POG予測に重要な特徴量を生成する（デビュー前に入手可能な情報のみ）:
- 血統スコア（父馬の産駒EI、母父馬の産駒EI、母馬の獲得賞金）
- 生まれ月（早生まれほど有利）
- 親年齢（父・母の産駒時年齢）
- 調教師スコア（有力調教師の実績ベース）
- 祖父母年齢の集約統計量（平均・最小・散布度）
- 曾祖父母年齢の集約統計量（平均・最小・散布度）
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

        # 調教師・馬主・生産者スコア
        row["trainer_score"] = calc_trainer_score(horse.get("trainer", ""))
        row["owner_score"] = calc_owner_score(horse.get("owner", ""))
        row["breeder_score"] = calc_breeder_score(horse.get("breeder", ""))

        # 生まれ月（1-12、小さいほど有利）
        row["birth_month"] = get_birth_month(hid, birth_year)

        # 親年齢（父・母・母父の産駒時年齢）— 欠損はNaN（デフォルト値補完しない）
        sire_age = get_parent_age(hid, birth_year, "sire")
        dam_age = get_parent_age(hid, birth_year, "dam")
        dam_sire_age = get_parent_age(hid, birth_year, "dam_sire")
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

        # --- 集約特徴量（世代別の統計量） ---
        # 母馬賞金の対数変換
        row["dam_prize_log"] = np.log1p(row["dam_prize"])

        # 血統交互作用（父の種牡馬力 × 母の実績）
        row["sire_dam_interaction"] = row["sire_ei"] * row["dam_prize_log"]

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)
