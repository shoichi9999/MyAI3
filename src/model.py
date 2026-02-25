"""
POGスコアリング — ヒューリスティック方式。

デビュー前に入手可能な特徴量から、ドメイン知識ベースの
重み付けスコアを算出して馬をランク付けする。
重みパラメータは data/config/weights.json から読み込む。
"""

import json
import os

import numpy as np
import pandas as pd


def _load_weights() -> dict:
    """data/config/weights.json から重みパラメータを読み込む。"""
    path = "data/config/weights.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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
    W = _load_weights()

    score = pd.Series(0.0, index=df.index)

    # 性別ボーナス
    if "sex" in df.columns:
        sex = df["sex"].fillna(0.5)
        score += (sex - 0.5) * W.get("b_sex", 16.47)

    # 父EI（99パーセンタイル正規化）
    if "sire_ei" in df.columns:
        ei = df["sire_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * W.get("w_sire_ei", 0.065)

    # 母父EI（99パーセンタイル正規化）
    if "bms_ei" in df.columns:
        ei = df["bms_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * W.get("w_bms_ei", 0.0235)

    # 初年度種牡馬ボーナス（年内正規化: 初年度種牡馬間の相対順位でスコア付け）
    if "sire_prize" in df.columns and "sire_ei" in df.columns:
        is_first_crop = df["sire_ei"].fillna(0) == 0
        sire_prize_log = np.log1p(df["sire_prize"].fillna(0))
        fc_max = sire_prize_log[is_first_crop].max() if is_first_crop.any() else 0
        normalized = (sire_prize_log / fc_max) if fc_max > 0 else sire_prize_log * 0
        score += is_first_crop * normalized * W.get("w_first_crop", 0.455)

    # 母馬獲得賞金（対数 + 99パーセンタイル正規化）
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        cap = dp.quantile(0.99)
        if cap > 0:
            score += (dp.clip(upper=cap) / cap) * 100 * W.get("w_dam_prize", 0.036)

    # 調教師スコアボーナス
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * W.get("w_trainer", 0.270)

    # 馬主スコアボーナス
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * W.get("w_owner", 0.143)

    # 早生まれボーナス
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * W.get("b_early", 0.0)

    # 両親若齢ボーナス
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * W.get("b_parents_young", 0.0)
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * W.get("b_parents_young", 0.0) / 2

    # 産駒番号（初仔ペナルティ、2-4番仔ボーナス）
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -W.get("b_foal_penalty", 14.97),
                          np.where(fn <= 4, W.get("b_foal_bonus", 6.24), 0))

    # 種牡馬高齢ペナルティ（16歳超で年齢に応じた減点）
    if "sire_age" in df.columns:
        sa = df["sire_age"].fillna(12)
        score -= np.maximum(0, sa - 16) * W.get("b_sire_old", 0.0)

    # セリ価格ボーナス
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * W.get("b_sale_price", 0.0)

    # 母馬の繁殖入り年齢（若いほど良い = 良血馬ほど早く繁殖入り）
    if "dam_breeding_age" in df.columns:
        dba = df["dam_breeding_age"]
        base = W.get("dam_breed_base", 5.0)
        cap_val = W.get("dam_breed_cap", 2.0)
        penalty = W.get("dam_breed_penalty", 3.0)
        score += np.where(dba.isna(), 0, (base - dba).clip(-penalty, cap_val))

    # 生産牧場スコア
    if "breeder_score" in df.columns:
        bs = df["breeder_score"].fillna(50)
        score += (bs - 50) * W.get("w_breeder", 0.0)

    # 母-母父年齢差
    if "dam_bms_gap_small" in df.columns:
        score += df["dam_bms_gap_small"].fillna(0) * W.get("b_dam_bms_gap", 0.0)

    # 母馬総産駒数ボーナス（適度な繁殖実績: 3-6頭がスイートスポット）
    if "total_dam_foals" in df.columns:
        tdf = df["total_dam_foals"].fillna(0)
        score += np.where((tdf >= 3) & (tdf <= 6), W.get("b_dam_foals_sweet", 0.0), 0)

    # 父EI × 母賞金交互作用（99パーセンタイル正規化）
    if "sire_dam_interaction" in df.columns:
        inter = df["sire_dam_interaction"].fillna(0)
        cap = inter.quantile(0.99)
        if cap > 0:
            score += (inter.clip(upper=cap) / cap) * W.get("w_sire_dam_inter", 0.0)

    # 調教師 × 生産者コンボ（エリート連携ボーナス）
    if "trainer_breeder_combo" in df.columns:
        combo = df["trainer_breeder_combo"].fillna(2500)
        # 基準値: 50*50=2500（非エリート同士）
        score += np.maximum(0, combo - 2500) * W.get("w_trainer_breeder", 0.0)

    # 兄姉のクラシック実績ボーナス（時点制約済み: birth_year-2以前の結果のみ）
    if "sibling_classic" in df.columns:
        score += df["sibling_classic"].fillna(0) * W.get("b_sibling_classic", 0.0)

    # 種牡馬のクラシックTOP5輩出数（時点制約済み: birth_year-2以前の結果のみ）
    if "sire_classic_count" in df.columns:
        score += df["sire_classic_count"].fillna(0) * W.get("w_sire_classic", 0.0)

    # 輸入繁殖牝馬ボーナス（海外産の母馬 = dam_prize不明でも良血の可能性）
    if "imported_dam" in df.columns:
        score += df["imported_dam"].fillna(0) * W.get("b_imported_dam", 0.0)

    # 種牡馬勝率（EIとは異なる角度の品質指標）
    if "sire_win_rate" in df.columns:
        wr = df["sire_win_rate"].fillna(0)
        score += wr * 100 * W.get("w_sire_win_rate", 0.0)

    # 母父勝率
    if "bms_win_rate" in df.columns:
        wr = df["bms_win_rate"].fillna(0)
        score += wr * 100 * W.get("w_bms_win_rate", 0.0)

    # 種牡馬ランクスコア（リーディング上位ほど高い）
    if "sire_rank_score" in df.columns:
        rs = df["sire_rank_score"].fillna(0)
        cap = rs.max()
        if cap > 0:
            score += (rs / cap) * 100 * W.get("w_sire_rank", 0.0)

    return score
