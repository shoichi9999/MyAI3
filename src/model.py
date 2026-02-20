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

    血統スコア（父EI 40% + 母馬賞金 40% + 母父EI 20%）に
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

    # 父EI（正規化）
    if "sire_ei" in df.columns:
        ei = df["sire_ei"].fillna(0)
        max_ei = ei.max()
        if max_ei > 0:
            score += (ei / max_ei) * 100 * WEIGHT_SIRE_EI

    # 母父EI（正規化）
    if "bms_ei" in df.columns:
        ei = df["bms_ei"].fillna(0)
        max_ei = ei.max()
        if max_ei > 0:
            score += (ei / max_ei) * 100 * WEIGHT_BMS_EI

    # 調教師スコアボーナス
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * 0.2  # 50基準で加減算

    # 馬主スコアボーナス
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * 0.2

    # 母馬獲得賞金（対数正規化）
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        max_dp = dp.max()
        if max_dp > 0:
            score += (dp / max_dp) * 100 * WEIGHT_DAM_PRIZE

    # 早生まれボーナス（1-4月生まれ）
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * 5

    # 両親若齢ボーナス（13歳以下）
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * 5
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * 2.5

    # 母と母父の年齢差が小さいボーナス
    if "dam_bms_gap_small" in df.columns:
        score += df["dam_bms_gap_small"].fillna(0) * 3

    # セリ価格ボーナス（高額馬ほど有利）
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * 8

    # 産駒番号（初仔は不利、2-4番仔がスイートスポット）
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -3, np.where(fn <= 4, 2, 0))

    # 生産牧場ボーナス
    if "breeder_score" in df.columns:
        bs = df["breeder_score"].fillna(50)
        score += (bs - 50) * 0.3

    return score
