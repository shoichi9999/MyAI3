"""
特徴量エンジニアリングモジュール。

POG予測に重要な特徴量を生成する:
- 血統スコア（父馬の産駒EI、母父馬の産駒EI、母馬の獲得賞金）
- 調教師スコア（2歳戦・クラシック実績）
- セリ価格
- 戦績から算出される能力指標
- 馬体重・成長曲線
"""

import json
import os
import re
from typing import Optional

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


# ヒューリスティックスコアの重み配分（バックテスト最良の E2 構成）
WEIGHT_SIRE_EI = 0.25
WEIGHT_DAM_PRIZE = 0.30
WEIGHT_BMS_EI = 0.15
WEIGHT_TRAINER = 0.30


def parse_prize_money(prize_str: str) -> float:
    """賞金文字列をfloat（万円単位）に変換する。"""
    if not prize_str or prize_str == "0":
        return 0.0
    cleaned = re.sub(r"[,\s万円]", "", str(prize_str))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_sale_price(price_str: str) -> float:
    """セリ取引価格の文字列をfloat（万円単位）に変換する。"""
    if not price_str or price_str == "-":
        return 0.0
    cleaned = re.sub(r"[,\s万円（税込）]", "", str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_horse_weight(weight_str: str) -> tuple[float, float]:
    """馬体重文字列を(体重, 増減)のタプルに変換する。"""
    if not weight_str:
        return (0.0, 0.0)
    match = re.match(r"(\d+)\(([+-]?\d+)\)", str(weight_str))
    if match:
        return (float(match.group(1)), float(match.group(2)))
    try:
        return (float(weight_str), 0.0)
    except ValueError:
        return (0.0, 0.0)


def parse_distance(dist_str: str) -> int:
    """距離文字列から数値（メートル）を抽出する。"""
    nums = re.findall(r"\d+", str(dist_str))
    if nums:
        return int(nums[-1])
    return 0


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


def calc_race_performance_features(results_df: pd.DataFrame, horse_id: str) -> dict:
    """戦績から能力指標を算出する。"""
    feats = {"horse_id": horse_id}

    if results_df.empty:
        feats.update({
            "num_races": 0, "num_wins": 0, "win_rate": 0.0, "top3_rate": 0.0,
            "total_earned": 0.0, "avg_finish": 0.0, "best_finish": 0,
            "avg_odds": 0.0, "max_distance": 0, "latest_weight": 0.0,
            "weight_trend": 0.0, "graded_race_wins": 0, "speed_rating": 0.0,
        })
        return feats

    horse_data = results_df[results_df["horse_id"] == horse_id].copy()
    if horse_data.empty:
        horse_data = results_df.copy()

    horse_data["finish_num"] = pd.to_numeric(
        horse_data["finish_position"], errors="coerce"
    )
    horse_data = horse_data.dropna(subset=["finish_num"])
    num_races = len(horse_data)
    feats["num_races"] = num_races

    if num_races == 0:
        feats.update({
            "num_wins": 0, "win_rate": 0.0, "top3_rate": 0.0,
            "total_earned": 0.0, "avg_finish": 0.0, "best_finish": 0,
            "avg_odds": 0.0, "max_distance": 0, "latest_weight": 0.0,
            "weight_trend": 0.0, "graded_race_wins": 0, "speed_rating": 0.0,
        })
        return feats

    num_wins = int((horse_data["finish_num"] == 1).sum())
    feats["num_wins"] = num_wins
    feats["win_rate"] = num_wins / num_races
    top3 = int((horse_data["finish_num"] <= 3).sum())
    feats["top3_rate"] = top3 / num_races

    if "prize" in horse_data.columns:
        horse_data["prize_val"] = horse_data["prize"].apply(parse_prize_money)
        feats["total_earned"] = horse_data["prize_val"].sum()
    else:
        feats["total_earned"] = 0.0

    feats["avg_finish"] = horse_data["finish_num"].mean()
    feats["best_finish"] = int(horse_data["finish_num"].min())

    if "odds" in horse_data.columns:
        odds_vals = pd.to_numeric(horse_data["odds"], errors="coerce")
        feats["avg_odds"] = odds_vals.mean() if not odds_vals.isna().all() else 0.0
    else:
        feats["avg_odds"] = 0.0

    if "distance" in horse_data.columns:
        distances = horse_data["distance"].apply(parse_distance)
        feats["max_distance"] = int(distances.max()) if not distances.empty else 0
    else:
        feats["max_distance"] = 0

    if "horse_weight" in horse_data.columns:
        weights = horse_data["horse_weight"].apply(lambda x: parse_horse_weight(x)[0])
        weights = weights[weights > 0]
        if not weights.empty:
            feats["latest_weight"] = weights.iloc[-1]
            feats["weight_trend"] = weights.iloc[-1] - weights.iloc[0] if len(weights) >= 2 else 0.0
        else:
            feats["latest_weight"] = 0.0
            feats["weight_trend"] = 0.0
    else:
        feats["latest_weight"] = 0.0
        feats["weight_trend"] = 0.0

    if "race_name" in horse_data.columns:
        graded = horse_data[
            horse_data["race_name"].str.contains(r"G[123I]|重賞|OP", na=False)
            & (horse_data["finish_num"] == 1)
        ]
        feats["graded_race_wins"] = len(graded)
    else:
        feats["graded_race_wins"] = 0

    feats["speed_rating"] = _calc_speed_rating(horse_data)
    return feats


def _calc_speed_rating(horse_data: pd.DataFrame) -> float:
    """簡易スピードレーティングを算出する。"""
    if "time" not in horse_data.columns or "distance" not in horse_data.columns:
        return 0.0

    ratings = []
    for _, row in horse_data.iterrows():
        time_str = str(row.get("time", ""))
        dist = parse_distance(str(row.get("distance", "")))
        if not time_str or dist == 0:
            continue
        seconds = _time_to_seconds(time_str)
        if seconds <= 0:
            continue
        base_time = (dist / 200) * 12.0
        rating = (base_time - seconds) * (2400 / dist) * 10 + 50
        ratings.append(rating)

    return np.mean(ratings) if ratings else 0.0


def _time_to_seconds(time_str: str) -> float:
    """タイム文字列(例: "1:35.2")を秒に変換する。"""
    match = re.match(r"(\d+):(\d+)\.(\d+)", str(time_str))
    if match:
        return int(match.group(1)) * 60 + int(match.group(2)) + int(match.group(3)) / 10
    match2 = re.match(r"(\d+)\.(\d+)", str(time_str))
    if match2:
        return int(match2.group(1)) + int(match2.group(2)) / 10
    return 0.0


def build_feature_matrix(
    horses_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    全馬の特徴量マトリクスを構築する。

    産駒成績ベースの血統スコア（父EI、母父EI、母馬獲得賞金）と
    調教師スコアを使って予測に使える特徴量を生成する。
    """
    feature_rows = []

    # 母馬賞金の中央値（賞金0の馬への補完用）
    dam_prizes_list = [v for v in DAM_PRIZES.values() if v > 0]
    dam_median = np.median(dam_prizes_list) if dam_prizes_list else 0.0

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

        # 調教師スコア
        row["trainer_score"] = calc_trainer_score(horse.get("trainer", ""))

        # セリ価格
        row["sale_price"] = parse_sale_price(horse.get("sale_price", ""))

        # 戦績ベースの特徴量
        if not results_df.empty:
            horse_results = results_df[results_df["horse_id"] == hid]
            perf = calc_race_performance_features(horse_results, hid)
        else:
            perf = calc_race_performance_features(pd.DataFrame(), hid)

        row.update({k: v for k, v in perf.items() if k != "horse_id"})

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)
