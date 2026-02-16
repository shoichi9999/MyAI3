"""
特徴量エンジニアリングモジュール。

POG予測に重要な特徴量を生成する:
- 血統スコア（父馬の産駒成績、母父馬の実績）
- 調教師スコア（2歳戦・クラシック実績）
- セリ価格
- 戦績から算出される能力指標
- 馬体重・成長曲線
"""

import re
from typing import Optional

import numpy as np
import pandas as pd


# POGで重要な種牡馬とそのスコア（過去の実績ベース）
# 実運用時はscraper.pyで取得した産駒成績から動的に算出する
ELITE_SIRES = {
    "ディープインパクト": 95,
    "キングカメハメハ": 88,
    "ロードカナロア": 90,
    "ハーツクライ": 85,
    "エピファネイア": 88,
    "ドゥラメンテ": 90,
    "キタサンブラック": 87,
    "モーリス": 83,
    "サトノダイヤモンド": 78,
    "スワーヴリチャード": 82,
    "コントレイル": 85,
    "シャフリヤール": 75,
    "イクイノックス": 80,
    "ドレフォン": 80,
    "サートゥルナーリア": 82,
    "レイデオロ": 78,
    "リアルスティール": 76,
    "ブリックスアンドモルタル": 79,
    "ニューイヤーズデイ": 77,
    "マインドユアビスケッツ": 74,
}

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
}


def parse_prize_money(prize_str: str) -> float:
    """賞金文字列をfloat（万円単位）に変換する。"""
    if not prize_str or prize_str == "0":
        return 0.0
    # カンマ、スペース除去
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


def calc_sire_score(sire_name: str) -> float:
    """種牡馬スコアを返す。未知の種牡馬はデフォルト値。"""
    if not sire_name:
        return 50.0
    return ELITE_SIRES.get(sire_name, 50.0)


def calc_trainer_score(trainer_name: str) -> float:
    """調教師スコアを返す。"""
    if not trainer_name:
        return 50.0
    # 部分一致で検索（所属厩舎名が含まれる場合があるため）
    for key, score in ELITE_TRAINERS.items():
        if key in str(trainer_name):
            return score
    return 50.0


def calc_race_performance_features(results_df: pd.DataFrame, horse_id: str) -> dict:
    """
    戦績から能力指標を算出する。

    Parameters
    ----------
    results_df : pd.DataFrame
        対象馬の戦績データ
    horse_id : str
        馬ID

    Returns
    -------
    dict
        算出された特徴量
    """
    feats = {"horse_id": horse_id}

    if results_df.empty:
        feats.update({
            "num_races": 0,
            "num_wins": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "total_earned": 0.0,
            "avg_finish": 0.0,
            "best_finish": 0,
            "avg_odds": 0.0,
            "max_distance": 0,
            "latest_weight": 0.0,
            "weight_trend": 0.0,
            "graded_race_wins": 0,
            "speed_rating": 0.0,
        })
        return feats

    horse_data = results_df[results_df["horse_id"] == horse_id].copy()
    if horse_data.empty:
        horse_data = results_df.copy()

    # 着順を数値化
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

    # 勝利数・勝率
    num_wins = int((horse_data["finish_num"] == 1).sum())
    feats["num_wins"] = num_wins
    feats["win_rate"] = num_wins / num_races

    # 3着以内率
    top3 = int((horse_data["finish_num"] <= 3).sum())
    feats["top3_rate"] = top3 / num_races

    # 獲得賞金合計
    if "prize" in horse_data.columns:
        horse_data["prize_val"] = horse_data["prize"].apply(parse_prize_money)
        feats["total_earned"] = horse_data["prize_val"].sum()
    else:
        feats["total_earned"] = 0.0

    # 平均着順
    feats["avg_finish"] = horse_data["finish_num"].mean()
    feats["best_finish"] = int(horse_data["finish_num"].min())

    # 平均オッズ（人気の指標）
    if "odds" in horse_data.columns:
        odds_vals = pd.to_numeric(horse_data["odds"], errors="coerce")
        feats["avg_odds"] = odds_vals.mean() if not odds_vals.isna().all() else 0.0
    else:
        feats["avg_odds"] = 0.0

    # 最大距離（スタミナの指標）
    if "distance" in horse_data.columns:
        distances = horse_data["distance"].apply(parse_distance)
        feats["max_distance"] = int(distances.max()) if not distances.empty else 0
    else:
        feats["max_distance"] = 0

    # 馬体重関連
    if "horse_weight" in horse_data.columns:
        weights = horse_data["horse_weight"].apply(
            lambda x: parse_horse_weight(x)[0]
        )
        weights = weights[weights > 0]
        if not weights.empty:
            feats["latest_weight"] = weights.iloc[-1]
            if len(weights) >= 2:
                feats["weight_trend"] = weights.iloc[-1] - weights.iloc[0]
            else:
                feats["weight_trend"] = 0.0
        else:
            feats["latest_weight"] = 0.0
            feats["weight_trend"] = 0.0
    else:
        feats["latest_weight"] = 0.0
        feats["weight_trend"] = 0.0

    # 重賞勝ち数
    if "race_name" in horse_data.columns:
        graded = horse_data[
            horse_data["race_name"].str.contains(r"G[123I]|重賞|OP", na=False)
            & (horse_data["finish_num"] == 1)
        ]
        feats["graded_race_wins"] = len(graded)
    else:
        feats["graded_race_wins"] = 0

    # スピードレーティング（簡易版）
    # タイムと距離から算出
    feats["speed_rating"] = _calc_speed_rating(horse_data)

    return feats


def _calc_speed_rating(horse_data: pd.DataFrame) -> float:
    """
    簡易スピードレーティングを算出する。
    基準タイムとの差分を距離で正規化して評価する。
    """
    if "time" not in horse_data.columns or "distance" not in horse_data.columns:
        return 0.0

    ratings = []
    for _, row in horse_data.iterrows():
        time_str = str(row.get("time", ""))
        dist = parse_distance(str(row.get("distance", "")))
        if not time_str or dist == 0:
            continue

        # タイムを秒に変換
        seconds = _time_to_seconds(time_str)
        if seconds <= 0:
            continue

        # 基準タイム（1ハロン=200m あたり12秒を基準）
        base_time = (dist / 200) * 12.0
        # レーティング = 基準タイムとの差 × 距離補正
        rating = (base_time - seconds) * (2400 / dist) * 10 + 50
        ratings.append(rating)

    return np.mean(ratings) if ratings else 0.0


def _time_to_seconds(time_str: str) -> float:
    """タイム文字列(例: "1:35.2")を秒に変換する。"""
    match = re.match(r"(\d+):(\d+)\.(\d+)", str(time_str))
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        tenths = int(match.group(3))
        return minutes * 60 + seconds + tenths / 10
    match2 = re.match(r"(\d+)\.(\d+)", str(time_str))
    if match2:
        seconds = int(match2.group(1))
        tenths = int(match2.group(2))
        return seconds + tenths / 10
    return 0.0


def build_feature_matrix(
    horses_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    全馬の特徴量マトリクスを構築する。

    Parameters
    ----------
    horses_df : pd.DataFrame
        馬プロフィール情報
    results_df : pd.DataFrame
        全戦績データ

    Returns
    -------
    pd.DataFrame
        特徴量マトリクス
    """
    feature_rows = []

    for _, horse in horses_df.iterrows():
        hid = horse.get("horse_id", "")
        row = {"horse_id": hid}

        # 基本情報
        row["horse_name"] = horse.get("horse_name", "")
        row["sex"] = 1 if horse.get("sex") == "牡" else (0 if horse.get("sex") == "牝" else 0.5)

        # 血統スコア
        row["sire_score"] = calc_sire_score(horse.get("sire", ""))
        row["dam_sire_score"] = calc_sire_score(horse.get("sire_of_dam", ""))

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
