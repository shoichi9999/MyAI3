"""
POGスコアリング — ヒューリスティック方式。

デビュー前に入手可能な特徴量から、ドメイン知識ベースの
重み付けスコアを算出して馬をランク付けする。
"""

import numpy as np
import pandas as pd


def heuristic_score(df: pd.DataFrame) -> pd.Series:
    """
    ヒューリスティックスコアを算出する。

    血統スコア（父EI 22.5% + 母馬賞金 7.5% + 母父EI 25%）に
    生まれ月・親年齢・市場評価等のボーナスを加算する。

    Parameters
    ----------
    df : pd.DataFrame
        build_feature_matrix() で生成した特徴量マトリクス

    Returns
    -------
    pd.Series
        各馬のスコア（高いほど有望）
    """
    from src.features import WEIGHT_SIRE_EI, WEIGHT_DAM_PRIZE, WEIGHT_BMS_EI

    score = pd.Series(0.0, index=df.index)

    # 父EI（99パーセンタイル正規化 — 外国種牡馬の外れ値EIで潰されるのを防ぐ）
    if "sire_ei" in df.columns:
        ei = df["sire_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * WEIGHT_SIRE_EI

    # 母父EI（99パーセンタイル正規化）
    if "bms_ei" in df.columns:
        ei = df["bms_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * WEIGHT_BMS_EI

    # 調教師スコアボーナス（50基準で加減算、限定的な寄与に抑える）
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * 0.05

    # 馬主スコアボーナス
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * 0.08

    # 初年度種牡馬ボーナス（産駒EIが未知の種牡馬に、自身の現役賞金で補正）
    # sire_ei == 0 を初年度とみなし、log1p(sire_prize) × 係数 を加算
    if "sire_prize" in df.columns and "sire_ei" in df.columns:
        is_first_crop = df["sire_ei"].fillna(0) == 0
        sire_prize_log = np.log1p(df["sire_prize"].fillna(0))
        # 係数0.8: 賞金1億(10000万)=log1p≈9.2→+7.4pt、賞金10億(100000万)=log1p≈11.5→+9.2pt
        score += is_first_crop * sire_prize_log * 0.8

    # 母馬獲得賞金（対数 + 99パーセンタイル正規化）
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        cap = dp.quantile(0.99)
        if cap > 0:
            score += (dp.clip(upper=cap) / cap) * 100 * WEIGHT_DAM_PRIZE

    # 早生まれボーナス（1-4月生まれ）
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * 8

    # 両親若齢ボーナス（13歳以下）
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * 8
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * 4

    # 母と母父の年齢差が小さいボーナス
    if "dam_bms_gap_small" in df.columns:
        score += df["dam_bms_gap_small"].fillna(0) * 5

    # セリ価格ボーナス（高額馬ほど有利）
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * 5

    # 産駒番号（初仔は不利、2-4番仔がスイートスポット）
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -5, np.where(fn <= 4, 3, 0))

    # 生産牧場ボーナス
    if "breeder_score" in df.columns:
        bs = df["breeder_score"].fillna(50)
        score += (bs - 50) * 0.15

    # 母馬の繁殖入り年齢（若いほど良い = 良血馬ほど早く繁殖入り）
    if "dam_breeding_age" in df.columns:
        dba = df["dam_breeding_age"]
        # 基準7歳、1歳若いごとに+1pt（上限+4）
        score += np.where(dba.isna(), 0, (7 - dba).clip(-2, 4))

    return score
