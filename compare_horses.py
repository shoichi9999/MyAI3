"""指定2頭の全特徴量とスコア内訳を完全分解して比較する。"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.features import build_feature_matrix
from src.model import _load_weights


def full_breakdown(df, race_type):
    """heuristic_score の全項を辞書で返す。"""
    W = _load_weights(race_type)
    cols = {}

    # 父EI
    ei = df["sire_ei"].fillna(0); cap = ei.quantile(0.99)
    cols["父EI"] = ((ei.clip(upper=cap) / cap) * 100 * W.get("w_sire_ei", 0)) if cap > 0 else 0
    # 母父EI
    bei = df["bms_ei"].fillna(0); bcap = bei.quantile(0.99)
    cols["母父EI"] = ((bei.clip(upper=bcap) / bcap) * 100 * W.get("w_bms_ei", 0)) if bcap > 0 else 0
    # 母馬賞金
    dp = np.log1p(df["dam_prize"].fillna(0)); dcap = dp.quantile(0.99)
    cols["母馬賞金"] = ((dp.clip(upper=dcap) / dcap) * 100 * W.get("w_dam_prize", 0)) if dcap > 0 else 0
    # 初年度種牡馬
    is_first = df["sire_ei"].fillna(0) == 0
    sp_log = np.log1p(df["sire_prize"].fillna(0))
    fc_max = sp_log[is_first].max() if is_first.any() else 0
    cols["初年度種牡馬"] = is_first * (sp_log / fc_max) * W.get("w_first_crop", 0) if fc_max > 0 else 0
    # 調教師・馬主・生産者
    cols["調教師"] = (df["trainer_score"].fillna(50) - 50) * W.get("w_trainer", 0)
    cols["馬主"] = (df["owner_score"].fillna(50) - 50) * W.get("w_owner", 0)
    cols["生産者"] = (df["breeder_score"].fillna(50) - 50) * W.get("w_breeder", 0)
    cols["早生まれ"] = df["early_born"].fillna(0) * W.get("b_early", 0)
    # 両親若齢
    if "both_parents_young" in df.columns:
        cols["両親若齢"] = df["both_parents_young"].fillna(0) * W.get("b_parents_young", 0)
    # 母-母父年齢差
    cols["母-母父年齢差"] = df["dam_bms_gap_small"].fillna(0) * W.get("b_dam_bms_gap", 0) if "dam_bms_gap_small" in df.columns else 0
    # セリ価格
    cols["セリ価格"] = df["sale_price_log"].fillna(0) * W.get("b_sale_price", 0) if "sale_price_log" in df.columns else 0
    # 産駒番号
    fn = df["foal_number"].fillna(3)
    cols["産駒番号"] = np.where(fn == 1, -W.get("b_foal_penalty", 0), np.where(fn <= 4, W.get("b_foal_bonus", 0), 0))
    # 種牡馬高齢
    sa = df["sire_age"].fillna(12)
    cols["種牡馬高齢ペナルティ"] = -np.maximum(0, sa - 16) * W.get("b_sire_old", 0)
    # 母馬繁殖入り年齢
    db = df["dam_breeding_age"].fillna(np.nan)
    db_notna = db.notna()
    breed_val = np.clip(W.get("dam_breed_base", 0) - db.fillna(0), -W.get("dam_breed_penalty", 0), W.get("dam_breed_cap", 0))
    cols["母馬繁殖入り年齢"] = db_notna.astype(int) * breed_val
    # 母馬産駒数(2-4番仔)
    tdf = df["total_dam_foals"].fillna(0)
    cols["母馬2-4番仔ボーナス"] = ((tdf >= 2) & (tdf <= 4)).astype(int) * W.get("b_dam_foals_sweet", 0)
    # 兄姉クラシック
    cols["兄姉クラシック"] = df["sibling_classic"].fillna(0) * W.get("b_sibling_classic", 0)
    # 母馬産駒品質
    cols["母馬産駒品質"] = df["dam_progeny_quality"].fillna(0) * W.get("b_dam_progeny_quality", 0)
    # 父クラシック実績/率/オークス率
    cols["父クラシック輩出数"] = df["sire_classic_count"].fillna(0) * W.get("w_sire_classic", 0)
    cols["父クラシック率"] = df["sire_classic_rate"].fillna(0) * W.get("w_sire_classic_rate", 0)
    cols["父オークス率"] = df["sire_oaks_rate"].fillna(0) * W.get("w_sire_oaks_rate", 0)
    # 母父クラシック輩出
    cols["母父クラシック輩出数"] = df["bms_classic_count"].fillna(0) * W.get("w_bms_classic", 0)
    # 父勝率/母父勝率
    cols["父勝率"] = df["sire_win_rate"].fillna(0) * 100 * W.get("w_sire_win_rate", 0)
    cols["母父勝率"] = df["bms_win_rate"].fillna(0) * 100 * W.get("w_bms_win_rate", 0)
    # ランクスコア（cap=max正規化）
    rs = df["sire_rank_score"].fillna(0); rcap = rs.max()
    cols["父ランクスコア"] = (rs / rcap * 100 * W.get("w_sire_rank", 0)) if rcap > 0 else 0
    brs = df["bms_rank_score"].fillna(0); brcap = brs.max()
    cols["母父ランクスコア"] = (brs / brcap * 100 * W.get("w_bms_rank", 0)) if brcap > 0 else 0
    # 産駒賞金/数
    sp = np.log1p(df["sire_progeny_prize"].fillna(0)); spcap = sp.quantile(0.99)
    cols["父産駒賞金"] = ((sp.clip(upper=spcap) / spcap) * 100 * W.get("w_sire_progeny_prize", 0)) if spcap > 0 else 0
    sr = np.log1p(df["sire_runners"].fillna(0)); srcap = sr.quantile(0.99)
    cols["父産駒数"] = ((sr.clip(upper=srcap) / srcap) * 100 * W.get("w_sire_runners", 0)) if srcap > 0 else 0
    br = np.log1p(df["bms_runners"].fillna(0)); brcap2 = br.quantile(0.99)
    cols["母父産駒数"] = ((br.clip(upper=brcap2) / brcap2) * 100 * W.get("w_bms_runners", 0)) if brcap2 > 0 else 0
    bp = np.log1p(df["bms_progeny_prize"].fillna(0)); bpcap = bp.quantile(0.99)
    cols["母父産駒賞金"] = ((bp.clip(upper=bpcap) / bpcap) * 100 * W.get("w_bms_progeny_prize", 0)) if bpcap > 0 else 0
    # 種牡馬2歳EI/早熟性/EIトレンド
    e2 = df["sire_2yo_ei"].fillna(0); e2cap = e2.quantile(0.99)
    cols["父2歳EI"] = ((e2.clip(upper=e2cap) / e2cap) * 100 * W.get("w_sire_2yo_ei", 0)) if e2cap > 0 else 0
    prec = df["sire_precocity"].fillna(0); pcap = prec.quantile(0.99)
    cols["父早熟性"] = ((prec.clip(upper=pcap) / pcap) * 100 * W.get("w_sire_precocity", 0)) if pcap > 0 else 0
    cols["父EIトレンド"] = df["sire_ei_trend"].fillna(0) * W.get("w_sire_ei_trend", 0)
    # 輸入繁殖牝馬
    cols["輸入繁殖牝馬"] = df["imported_dam"].fillna(0) * W.get("b_imported_dam", 0)
    # 交互作用
    sdi = df["sire_dam_interaction"].fillna(0); sdcap = sdi.quantile(0.99)
    cols["父×母賞金交互作用"] = ((sdi.clip(upper=sdcap) / sdcap) * W.get("w_sire_dam_inter", 0)) if sdcap > 0 else 0
    bdi = df["bms_dam_interaction"].fillna(0); bdcap = bdi.quantile(0.99)
    cols["母父×母賞金交互作用"] = ((bdi.clip(upper=bdcap) / bdcap) * W.get("w_bms_dam_inter", 0)) if bdcap > 0 else 0
    otc = df["owner_trainer_combo"].fillna(2500)
    cols["馬主×調教師コンボ"] = np.maximum(0, otc - 2500) * W.get("w_owner_trainer", 0)
    tbc = df["trainer_breeder_combo"].fillna(0)
    cols["調教師×生産者コンボ"] = tbc * W.get("w_trainer_breeder", 0)
    # 母馬高クラス
    cols["母馬高クラス"] = df["dam_high_class"].fillna(0) * W.get("b_dam_high_class", 0)
    # 配合理論
    cols["父Stayer×母父Miler"] = df["stayer_x_miler"].fillna(0) * W.get("w_stayer_miler", 0)
    cols["父中距離×母父スピード"] = df["mid_x_speed"].fillna(0) * W.get("w_mid_x_speed", 0)
    return pd.DataFrame(cols)


horses = pd.read_csv("data/horses_2024.csv")
feats = build_feature_matrix(horses, birth_year=2024, sex_filter="牝")
parts = full_breakdown(feats, "oaks")
total = parts.sum(axis=1)
feats["score"] = total

picks = ["ハナナ", "レジューノワール", "スプレマレイナ", "ブレイクガール"]
print(f'{"項目":<22}' + " ".join(f'{n:>14}' for n in picks))
print("-" * 90)

# 各馬の行インデックスを取得
idx_map = {n: feats[feats["horse_name"] == n].index[0] for n in picks if (feats["horse_name"] == n).any()}

rows = []
for col in parts.columns:
    vals = []
    for n in picks:
        if n in idx_map:
            vals.append(float(parts.iloc[idx_map[n]][col]))
        else:
            vals.append(None)
    rows.append((col, vals))

# 差が大きい順
rows.sort(key=lambda r: max(abs(v) for v in r[1] if v is not None), reverse=True)
for col, vals in rows:
    vs = " ".join(f'{v:>14.1f}' for v in vals)
    print(f'{col:<22}{vs}')

print("-" * 90)
print(f'{"合計":<22}' + " ".join(f'{total.iloc[idx_map[n]]:>14.1f}' for n in picks))
print(f'{"順位":<22}' + " ".join(f'{(feats.sort_values("score",ascending=False).reset_index().query("horse_name==@n").index[0]+1):>14}' for n in picks))

# メタ情報
print("\n=== メタ情報 ===")
print(f'{"特徴量":<22}' + " ".join(f'{n:>14}' for n in picks))
for key in ["sire_ei", "bms_ei", "dam_prize", "sire_age", "dam_age", "birth_month",
            "sibling_classic", "total_dam_foals", "foal_number", "imported_dam",
            "trainer_score", "owner_score", "breeder_score",
            "stayer_x_miler", "mid_x_speed"]:
    if key not in feats.columns:
        continue
    vs = " ".join(f'{feats.iloc[idx_map[n]][key]:>14}' for n in picks)
    print(f'{key:<22}{vs}')
