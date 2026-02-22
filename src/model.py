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

    # 性別ボーナス（TOP10最適化で最重要特徴量 — 牡馬のTOP入り率が圧倒的に高い）
    if "sex" in df.columns:
        sex = df["sex"].fillna(0.5)
        # 牡馬(1.0)=+8.23, セン(0.5)=0, 牝馬(0.0)=-8.23
        score += (sex - 0.5) * 16.47

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

    # 初年度種牡馬ボーナス（産駒EIが未知の種牡馬に、自身の現役賞金で補正）
    if "sire_prize" in df.columns and "sire_ei" in df.columns:
        is_first_crop = df["sire_ei"].fillna(0) == 0
        sire_prize_log = np.log1p(df["sire_prize"].fillna(0))
        score += is_first_crop * sire_prize_log * 0.355

    # 母馬獲得賞金（対数 + 99パーセンタイル正規化）
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        cap = dp.quantile(0.99)
        if cap > 0:
            score += (dp.clip(upper=cap) / cap) * 100 * WEIGHT_DAM_PRIZE

    # 調教師スコアボーナス
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * 0.270

    # 馬主スコアボーナス
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * 0.143

    # 早生まれボーナス（1-4月生まれ — TOP10では非常に重要）
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * 6.10

    # 両親若齢ボーナス（13歳以下）
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * 1.71
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * 0.86

    # 産駒番号（2-4番仔ボーナス、初仔ペナルティ — TOP10では初仔不利が顕著）
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -14.97,
                          np.where(fn <= 4, 6.24, 0))

    # セリ価格ボーナス
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * 1.94

    # 母馬の繁殖入り年齢（若いほど良い = 良血馬ほど早く繁殖入り）
    if "dam_breeding_age" in df.columns:
        dba = df["dam_breeding_age"]
        score += np.where(dba.isna(), 0, (1.84 - dba).clip(-7.25, 6.76))

    return score
