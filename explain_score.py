"""TOP10馬のスコア内訳を分解して説明する。

heuristic_score() の線形和を各項ごとに計算し、寄与度を可視化。
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.features import build_feature_matrix
from src.model import _load_weights, heuristic_score


def score_breakdown(df: pd.DataFrame, race_type: str = "derby") -> pd.DataFrame:
    """各馬のスコアを項ごとに分解した DataFrame を返す。"""
    W = _load_weights(race_type)
    parts = pd.DataFrame(index=df.index)

    # 父EI
    ei = df["sire_ei"].fillna(0)
    cap = ei.quantile(0.99)
    parts["父EI"] = (ei.clip(upper=cap) / cap) * 100 * W.get("w_sire_ei", 0) if cap > 0 else 0

    # 母父EI
    bei = df["bms_ei"].fillna(0)
    bcap = bei.quantile(0.99)
    parts["母父EI"] = (bei.clip(upper=bcap) / bcap) * 100 * W.get("w_bms_ei", 0) if bcap > 0 else 0

    # 母馬賞金
    dp = np.log1p(df["dam_prize"].fillna(0))
    dcap = dp.quantile(0.99)
    parts["母馬賞金"] = (dp.clip(upper=dcap) / dcap) * 100 * W.get("w_dam_prize", 0) if dcap > 0 else 0

    # 初年度種牡馬
    is_first = df["sire_ei"].fillna(0) == 0
    sp_log = np.log1p(df["sire_prize"].fillna(0))
    fc_max = sp_log[is_first].max() if is_first.any() else 0
    if fc_max > 0:
        parts["初年度種牡馬"] = is_first * (sp_log / fc_max) * W.get("w_first_crop", 0)
    else:
        parts["初年度種牡馬"] = 0

    # 調教師・馬主・生産者
    parts["調教師"] = (df["trainer_score"].fillna(50) - 50) * W.get("w_trainer", 0)
    parts["馬主"] = (df["owner_score"].fillna(50) - 50) * W.get("w_owner", 0)
    parts["生産者"] = (df["breeder_score"].fillna(50) - 50) * W.get("w_breeder", 0)

    # 早生まれ
    parts["早生まれ"] = df["early_born"].fillna(0) * W.get("b_early", 0)

    # 産駒番号
    fn = df["foal_number"].fillna(3)
    parts["産駒番号"] = np.where(fn == 1, -W.get("b_foal_penalty", 0),
                                  np.where(fn <= 4, W.get("b_foal_bonus", 0), 0))

    # 種牡馬高齢ペナルティ
    sa = df["sire_age"].fillna(12)
    parts["種牡馬高齢"] = -np.maximum(0, sa - 16) * W.get("b_sire_old", 0)

    # 兄姉クラシック
    parts["兄姉クラシック"] = df["sibling_classic"].fillna(0) * W.get("b_sibling_classic", 0)

    # 種牡馬クラシック輩出数
    parts["父クラシック実績"] = df["sire_classic_count"].fillna(0) * W.get("w_sire_classic", 0)

    # 種牡馬クラシック率
    parts["父クラシック率"] = df["sire_classic_rate"].fillna(0) * W.get("w_sire_classic_rate", 0)

    # 種牡馬EIトレンド
    parts["父EIトレンド"] = df["sire_ei_trend"].fillna(0) * W.get("w_sire_ei_trend", 0)

    # 父Stayer×母父Miler
    parts["父Stayer×母父Miler"] = df["stayer_x_miler"].fillna(0) * W.get("w_stayer_miler", 0)

    # 父中距離以上×母父スピード
    parts["父中距離以上×母父スピード"] = df["mid_x_speed"].fillna(0) * W.get("w_mid_x_speed", 0)

    # 輸入繁殖牝馬
    parts["輸入繁殖牝馬"] = df["imported_dam"].fillna(0) * W.get("b_imported_dam", 0)

    # 父×母交互作用
    sdi = df["sire_dam_interaction"].fillna(0)
    sdcap = sdi.quantile(0.99)
    if sdcap > 0:
        parts["父×母賞金交互作用"] = (sdi.clip(upper=sdcap) / sdcap) * 100 * W.get("w_sire_dam_inter", 0)
    else:
        parts["父×母賞金交互作用"] = 0

    return parts


def explain_horse(horses_df, race_type, picks_names):
    sex = "牡" if race_type == "derby" else "牝"
    feats = build_feature_matrix(horses_df, birth_year=2024, sex_filter=sex)
    # 正規のheuristic_scoreで順位を決める
    feats["score_total"] = heuristic_score(feats, race_type=race_type)
    # 分解は表示用
    parts = score_breakdown(feats, race_type)
    feats = feats.sort_values("score_total", ascending=False).reset_index(drop=True)
    feats["rank"] = feats.index + 1
    parts_idx = parts.reindex(feats.index)

    label = "ダービー（牡馬）" if race_type == "derby" else "オークス（牝馬）"
    print(f"\n{'='*60}\n  {label} TOP10 スコア内訳\n{'='*60}")

    for nm in picks_names:
        row = feats[feats["horse_name"] == nm]
        if row.empty:
            print(f"\n[{nm}] NOT FOUND")
            continue
        i = row.index[0]
        rank = int(row.iloc[0]["rank"])
        total_s = float(row.iloc[0]["score_total"])
        p = parts_idx.iloc[i].sort_values(ascending=False)
        # 上位寄与（絶対値大きい順）
        top_parts = p.reindex(p.abs().sort_values(ascending=False).index).head(8)
        top_parts = top_parts[top_parts.abs() > 0.5]
        meta = feats.iloc[i]
        print(f"\n[{rank}位] {nm}  (合計スコア {total_s:.1f})")
        print(f"  父={meta.get('sire_ei','?'):.2f}EI, 母父={meta.get('bms_ei','?'):.2f}EI, "
              f"母賞金={int(meta['dam_prize']):>6}, "
              f"Stayer×Miler={int(meta['stayer_x_miler'])}, "
              f"Mid×Speed={int(meta['mid_x_speed'])}")
        for k, v in top_parts.items():
            sign = "+" if v >= 0 else ""
            print(f"    {k:<28} {sign}{v:>7.1f}")


horses = pd.read_csv("data/horses_2024.csv")
derby_picks = [
    "マテンロウシヴァ", "ソブリオ", "ヴェルバーニア", "フェステグラウベ", "マスターズボンド",
    "ジャンゴッド", "ジャスティンカンヌ", "プロテウス", "エルドボルグ", "スタニスラス",
]
oaks_picks = [
    "ハナナ", "スプレマレイナ", "ブレイクガール", "サンタンジェロ", "ファジルノーア",
    "サンリットブーケ", "ルフトシュピール", "レジューノワール", "プリモシーンの2024", "アレオクール",
]
explain_horse(horses, "derby", derby_picks)
explain_horse(horses, "oaks", oaks_picks)
