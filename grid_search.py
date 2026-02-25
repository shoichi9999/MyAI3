"""
グリッドサーチ — ヒューリスティックスコアの重み最適化。

ダービー/オークスTOP5予測を最大化するようにパラメータを探索する。

使い方:
  python grid_search.py --years 2019
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from src.features import build_feature_matrix


# ------------------------------------------------------------------
# パラメータ化されたスコア関数
# ------------------------------------------------------------------

def parameterized_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """パラメータ辞書でスコアを計算する。"""
    score = pd.Series(0.0, index=df.index)

    # 性別ボーナス
    if "sex" in df.columns:
        sex = df["sex"].fillna(0.5)
        score += (sex - 0.5) * params.get("b_sex", 10)

    w_sire = params["w_sire_ei"]
    w_dam = params["w_dam_prize"]
    w_bms = params["w_bms_ei"]

    # 父EI
    if "sire_ei" in df.columns:
        ei = df["sire_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * w_sire

    # 母父EI
    if "bms_ei" in df.columns:
        ei = df["bms_ei"].fillna(0)
        cap = ei.quantile(0.99)
        if cap > 0:
            score += (ei.clip(upper=cap) / cap) * 100 * w_bms

    # 初年度種牡馬ボーナス（年内正規化）
    if "sire_prize" in df.columns and "sire_ei" in df.columns:
        is_first_crop = df["sire_ei"].fillna(0) == 0
        sire_prize_log = np.log1p(df["sire_prize"].fillna(0))
        fc_max = sire_prize_log[is_first_crop].max() if is_first_crop.any() else 0
        normalized = (sire_prize_log / fc_max) if fc_max > 0 else sire_prize_log * 0
        score += is_first_crop * normalized * params["w_first_crop"]

    # 母馬賞金
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        cap = dp.quantile(0.99)
        if cap > 0:
            score += (dp.clip(upper=cap) / cap) * 100 * w_dam

    # 調教師
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * params["w_trainer"]

    # 馬主
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * params["w_owner"]

    # 早生まれ
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * params["b_early"]

    # 両親若齢
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * params["b_parents_young"]
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * (params["b_parents_young"] / 2)

    # 母-母父年齢差
    if "dam_bms_gap_small" in df.columns:
        score += df["dam_bms_gap_small"].fillna(0) * params["b_dam_bms_gap"]

    # 種牡馬高齢ペナルティ
    if "sire_age" in df.columns:
        sa = df["sire_age"].fillna(12)
        score -= np.maximum(0, sa - 16) * params.get("b_sire_old", 0)

    # セリ価格
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * params["b_sale_price"]

    # 産駒番号
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -params["b_foal_penalty"],
                          np.where(fn <= 4, params["b_foal_bonus"], 0))

    # 生産牧場
    if "breeder_score" in df.columns:
        bs = df["breeder_score"].fillna(50)
        score += (bs - 50) * params["w_breeder"]

    # 母馬の繁殖入り年齢
    if "dam_breeding_age" in df.columns:
        dba = df["dam_breeding_age"]
        score += np.where(dba.isna(), 0,
                          (params["dam_breed_base"] - dba).clip(
                              -params["dam_breed_penalty"], params["dam_breed_cap"]))

    # 母馬総産駒数ボーナス（3-6頭がスイートスポット）
    if "total_dam_foals" in df.columns:
        tdf = df["total_dam_foals"].fillna(0)
        score += np.where((tdf >= 3) & (tdf <= 6), params.get("b_dam_foals_sweet", 0), 0)

    # 父EI × 母賞金交互作用
    if "sire_dam_interaction" in df.columns:
        inter = df["sire_dam_interaction"].fillna(0)
        cap = inter.quantile(0.99)
        if cap > 0:
            score += (inter.clip(upper=cap) / cap) * params.get("w_sire_dam_inter", 0)

    # 調教師 × 生産者コンボ
    if "trainer_breeder_combo" in df.columns:
        combo = df["trainer_breeder_combo"].fillna(2500)
        score += np.maximum(0, combo - 2500) * params.get("w_trainer_breeder", 0)

    # 兄姉のクラシック実績ボーナス（時点制約済み）
    if "sibling_classic" in df.columns:
        score += df["sibling_classic"].fillna(0) * params.get("b_sibling_classic", 0)

    # 種牡馬クラシックTOP5輩出数（時点制約済み）
    if "sire_classic_count" in df.columns:
        score += df["sire_classic_count"].fillna(0) * params.get("w_sire_classic", 0)

    # 輸入繁殖牝馬ボーナス
    if "imported_dam" in df.columns:
        score += df["imported_dam"].fillna(0) * params.get("b_imported_dam", 0)

    return score


# ------------------------------------------------------------------
# クラシック結果データ
# ------------------------------------------------------------------

def _load_classic_results() -> dict:
    path = "data/classic_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_CLASSIC = _load_classic_results()


def _get_classic_ids(birth_year: int) -> tuple[set, set]:
    """(derby_top5_ids, oaks_top5_ids) を返す。"""
    derby = set(_CLASSIC.get("derby", {}).get(str(birth_year), []))
    oaks = set(_CLASSIC.get("oaks", {}).get(str(birth_year), []))
    return derby, oaks


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_single(df: pd.DataFrame, params: dict, birth_year: int = None) -> dict:
    """1年度のデータでダービー/オークスTOP5予測を評価する。"""
    scores = parameterized_score(df, params).values
    horse_ids = df["horse_id"].astype(str).values
    sex = df["sex"].values

    if birth_year is None:
        birth_year = df.get("_birth_year", 2020)

    derby_top5, oaks_top5 = _get_classic_ids(birth_year)
    classic_top10 = derby_top5 | oaks_top5

    results = {}

    # 全体: 予測TOP N にクラシック馬が何頭入るか
    for n in [10, 20, 30, 50, 100]:
        pred_idx = np.argsort(-scores)[:n]
        pred_ids = set(horse_ids[pred_idx])
        results[f"top{n}_total"] = len(pred_ids & classic_top10)

    # 牡馬内でダービー評価
    male_mask = sex == 1
    if male_mask.any() and derby_top5:
        male_scores = scores[male_mask]
        male_ids = horse_ids[male_mask]
        for n in [10, 20, 30]:
            pred_idx = np.argsort(-male_scores)[:n]
            pred_ids = set(male_ids[pred_idx])
            results[f"male_top{n}_derby"] = len(pred_ids & derby_top5)

    # 牝馬内でオークス評価
    female_mask = sex == 0
    if female_mask.any() and oaks_top5:
        female_scores = scores[female_mask]
        female_ids = horse_ids[female_mask]
        for n in [10, 20, 30]:
            pred_idx = np.argsort(-female_scores)[:n]
            pred_ids = set(female_ids[pred_idx])
            results[f"female_top{n}_oaks"] = len(pred_ids & oaks_top5)

    return results


def composite_score(metrics: dict, objective: str = "classic") -> float:
    """複合スコア（全体TOP10 + 牡馬ダービー + 牝馬オークス）。"""
    top10 = metrics.get("top10_total", 0)
    top20 = metrics.get("top20_total", 0)
    top30 = metrics.get("top30_total", 0)

    # 牡馬内ダービーヒット / 牝馬内オークスヒット
    m_d10 = metrics.get("male_top10_derby", 0)
    m_d20 = metrics.get("male_top20_derby", 0)
    f_o10 = metrics.get("female_top10_oaks", 0)
    f_o20 = metrics.get("female_top20_oaks", 0)

    return (
        top10 * 100.0   # 全体TOP10ヒットが最重要
        + top20 * 10.0  # TOP20
        + top30 * 3.0   # TOP30
        + m_d10 * 80.0  # 牡馬TOP10内ダービーヒット
        + m_d20 * 8.0   # 牡馬TOP20内ダービーヒット
        + f_o10 * 80.0  # 牝馬TOP10内オークスヒット
        + f_o20 * 8.0   # 牝馬TOP20内オークスヒット
    )


def cv_score(all_data: dict, params: dict) -> float:
    """全年度の composite_score 平均を返す。"""
    scores = []
    for year, df in all_data.items():
        m = evaluate_single(df, params, birth_year=year)
        scores.append(composite_score(m))
    return np.mean(scores)


# ------------------------------------------------------------------
# データ読み込み
# ------------------------------------------------------------------

def load_year(year: int) -> pd.DataFrame | None:
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        return None
    # クラシック結果がない年はスキップ
    derby, oaks = _get_classic_ids(year)
    if not derby and not oaks:
        return None
    horses = pd.read_csv(csv_path)
    features = build_feature_matrix(horses, birth_year=year)
    return features


# ------------------------------------------------------------------
# グリッドサーチ
# ------------------------------------------------------------------

def grid_search(years, objective="balanced"):
    global _OBJECTIVE
    _OBJECTIVE = objective
    print(f"=== データ読み込み === (目的関数: {objective})")
    all_data = {}
    for y in years:
        df = load_year(y)
        if df is not None:
            all_data[y] = df
            print(f"  {y}年: {len(df)}頭")
        else:
            print(f"  {y}年: データなし（スキップ）")

    if not all_data:
        print("[ERROR] データが見つかりません")
        return

    print(f"\n  利用年度: {sorted(all_data.keys())} ({len(all_data)}年分)")

    # 現行パラメータ（data/config/weights.json から読み込み）
    from src.model import _load_weights
    _w = _load_weights()
    current_params = {
        "b_sex": _w.get("b_sex", 16.47),
        "w_sire_ei": _w.get("w_sire_ei", 0.065),
        "w_dam_prize": _w.get("w_dam_prize", 0.036),
        "w_bms_ei": _w.get("w_bms_ei", 0.0235),
        "w_first_crop": _w.get("w_first_crop", 0.455),
        "b_early": _w.get("b_early", 0.0),
        "b_parents_young": _w.get("b_parents_young", 0.0),
        "b_dam_bms_gap": _w.get("b_dam_bms_gap", 0.0),
        "b_sale_price": _w.get("b_sale_price", 0.0),
        "b_foal_penalty": _w.get("b_foal_penalty", 14.97),
        "b_foal_bonus": _w.get("b_foal_bonus", 6.24),
        "w_trainer": _w.get("w_trainer", 0.270),
        "w_owner": _w.get("w_owner", 0.143),
        "w_breeder": _w.get("w_breeder", 0.0),
        "dam_breed_base": _w.get("dam_breed_base", 5.0),
        "dam_breed_cap": _w.get("dam_breed_cap", 2.0),
        "dam_breed_penalty": _w.get("dam_breed_penalty", 3.0),
        "b_sire_old": _w.get("b_sire_old", 0.0),
        "b_dam_foals_sweet": _w.get("b_dam_foals_sweet", 0.0),
        "w_sire_dam_inter": _w.get("w_sire_dam_inter", 0.0),
        "w_trainer_breeder": _w.get("w_trainer_breeder", 0.0),
        "b_sibling_classic": _w.get("b_sibling_classic", 0.0),
        "w_sire_classic": _w.get("w_sire_classic", 0.0),
        "b_imported_dam": _w.get("b_imported_dam", 0.0),
    }

    current_cv = cv_score(all_data, current_params)
    print(f"\n  現行パラメータのCVスコア: {current_cv:.2f}")
    for y, df in sorted(all_data.items()):
        m = evaluate_single(df, current_params, birth_year=y)
        t10 = m.get("top10_total", 0)
        t20 = m.get("top20_total", 0)
        t30 = m.get("top30_total", 0)
        print(f"    {y}年: TOP10={t10}/10  TOP20={t20}/10  TOP30={t30}/10")

    best_score = current_cv
    best_params = dict(current_params)

    if objective == "top10":
        # TOP10最適化: 全パラメータ同時ランダム探索（局所最適回避）
        best_params, best_score = _random_search_top10(all_data, current_params, best_score)
    else:
        # balanced: 従来の段階的グリッドサーチ
        best_params, best_score = _staged_grid_search(all_data, dict(current_params), best_score)

    # ------------------------------------------------------------------
    # 最終結果
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  最適パラメータ")
    print("=" * 70)
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")
    print(f"\n  CVスコア: {best_score:.2f} (現行: {current_cv:.2f}, 差: {best_score - current_cv:+.2f})")

    # weights.json に書き戻し
    import json as _json
    weights_path = "data/config/weights.json"
    save_weights = dict(best_params)
    save_weights["_comment"] = "グリッドサーチ自動更新"
    with open(weights_path, "w", encoding="utf-8") as _f:
        _json.dump(save_weights, _f, ensure_ascii=False, indent=2)
    print(f"\n  → {weights_path} に保存しました")

    # 変更点
    print(f"\n--- 変更点 ---")
    changed = False
    for k in sorted(current_params):
        if current_params[k] != best_params[k]:
            print(f"  {k}: {current_params[k]} → {best_params[k]}")
            changed = True
    if not changed:
        print("  （変更なし — 現行パラメータが最適）")

    # 年度別詳細
    print(f"\n--- 年度別パフォーマンス比較 ---")
    print(f"{'年':>6} | {'指標':>12} | {'現行':>6} | {'最適化':>6} | {'差':>6}")
    print("-" * 55)
    for y in sorted(all_data.keys()):
        m_old = evaluate_single(all_data[y], current_params, birth_year=y)
        m_new = evaluate_single(all_data[y], best_params, birth_year=y)
        t10_old = m_old.get("top10_total", 0)
        t10_new = m_new.get("top10_total", 0)
        t20_old = m_old.get("top20_total", 0)
        t20_new = m_new.get("top20_total", 0)
        t30_old = m_old.get("top30_total", 0)
        t30_new = m_new.get("top30_total", 0)
        md10_old = m_old.get("male_top10_derby", 0)
        md10_new = m_new.get("male_top10_derby", 0)
        fo10_old = m_old.get("female_top10_oaks", 0)
        fo10_new = m_new.get("female_top10_oaks", 0)
        print(f"  {y} | {'全体TOP10':>12} | {t10_old:>4}/10 | {t10_new:>4}/10 | {t10_new-t10_old:>+5}")
        print(f"       | {'全体TOP20':>12} | {t20_old:>4}/10 | {t20_new:>4}/10 | {t20_new-t20_old:>+5}")
        print(f"       | {'全体TOP30':>12} | {t30_old:>4}/10 | {t30_new:>4}/10 | {t30_new-t30_old:>+5}")
        print(f"       | {'牡TOP10Derby':>12} | {md10_old:>4}/5  | {md10_new:>4}/5  | {md10_new-md10_old:>+5}")
        print(f"       | {'牝TOP10Oaks':>12} | {fo10_old:>4}/5  | {fo10_new:>4}/5  | {fo10_new-fo10_old:>+5}")
        print("-" * 55)

    return best_params


def _precompute_arrays(all_data):
    """DataFrameから高速評価用のnumpy配列を事前計算する。"""
    precomputed = {}
    for year, df in all_data.items():
        n = len(df)
        d = {}
        # 性別
        d["sex_centered"] = (df["sex"].fillna(0.5) - 0.5).values if "sex" in df.columns else np.zeros(n)
        # 父EI（正規化済み）
        if "sire_ei" in df.columns:
            ei = df["sire_ei"].fillna(0).values
            cap = np.percentile(ei, 99)
            d["sire_ei_norm"] = (np.clip(ei, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_ei_norm"] = np.zeros(n)
        # 母父EI（正規化済み）
        if "bms_ei" in df.columns:
            ei = df["bms_ei"].fillna(0).values
            cap = np.percentile(ei, 99)
            d["bms_ei_norm"] = (np.clip(ei, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["bms_ei_norm"] = np.zeros(n)
        # 初年度種牡馬ボーナス（年内正規化）
        if "sire_prize" in df.columns and "sire_ei" in df.columns:
            is_first = (df["sire_ei"].fillna(0) == 0).values.astype(float)
            sp_log = np.log1p(df["sire_prize"].fillna(0).values)
            fc_mask = is_first.astype(bool)
            fc_max = sp_log[fc_mask].max() if fc_mask.any() else 1.0
            normalized = (sp_log / fc_max) if fc_max > 0 else np.zeros(n)
            d["first_crop_val"] = is_first * normalized
        else:
            d["first_crop_val"] = np.zeros(n)
        # 母馬賞金（対数正規化）
        if "dam_prize" in df.columns:
            dp = np.log1p(df["dam_prize"].fillna(0).values)
            cap = np.percentile(dp, 99)
            d["dam_prize_norm"] = (np.clip(dp, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["dam_prize_norm"] = np.zeros(n)
        # 調教師・馬主・牧場（中心化済み）
        d["trainer_centered"] = (df["trainer_score"].fillna(50) - 50).values if "trainer_score" in df.columns else np.zeros(n)
        d["owner_centered"] = (df["owner_score"].fillna(50) - 50).values if "owner_score" in df.columns else np.zeros(n)
        d["breeder_centered"] = (df["breeder_score"].fillna(50) - 50).values if "breeder_score" in df.columns else np.zeros(n)
        # 早生まれ
        d["early_born"] = df["early_born"].fillna(0).values if "early_born" in df.columns else np.zeros(n)
        # 両親若齢
        if "both_parents_young" in df.columns:
            d["parents_young"] = df["both_parents_young"].fillna(0).values
        elif "sire_young" in df.columns and "dam_young" in df.columns:
            d["parents_young_half"] = (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)).values
        else:
            d["parents_young"] = np.zeros(n)
        # 母-母父年齢差
        d["dam_bms_gap"] = df["dam_bms_gap_small"].fillna(0).values if "dam_bms_gap_small" in df.columns else np.zeros(n)
        # 種牡馬高齢ペナルティ（16歳超の超過年数）
        if "sire_age" in df.columns:
            sa = df["sire_age"].fillna(12).values
            d["sire_old_excess"] = np.maximum(0, sa - 16)
        else:
            d["sire_old_excess"] = np.zeros(n)
        # セリ価格（正規化）
        if "sale_price_log" in df.columns:
            sp = df["sale_price_log"].fillna(0).values
            max_sp = sp.max()
            d["sale_price_norm"] = (sp / max_sp) if max_sp > 0 else np.zeros(n)
        else:
            d["sale_price_norm"] = np.zeros(n)
        # 産駒番号（事前マスク）
        if "foal_number" in df.columns:
            fn = df["foal_number"].fillna(3).values
            d["is_first_foal"] = (fn == 1).astype(float)
            d["is_good_foal"] = ((fn >= 2) & (fn <= 4)).astype(float)
        else:
            d["is_first_foal"] = np.zeros(n)
            d["is_good_foal"] = np.zeros(n)
        # 繁殖入り年齢
        if "dam_breeding_age" in df.columns:
            dba = df["dam_breeding_age"].values
            d["dam_breed_notna"] = (~np.isnan(dba)).astype(float)
            d["dam_breed_age"] = np.nan_to_num(dba, nan=0.0)
        else:
            d["dam_breed_notna"] = np.zeros(n)
            d["dam_breed_age"] = np.zeros(n)
        # 母馬総産駒数スイートスポット（3-6頭）
        if "total_dam_foals" in df.columns:
            tdf = df["total_dam_foals"].fillna(0).values
            d["dam_foals_sweet"] = ((tdf >= 3) & (tdf <= 6)).astype(float)
        else:
            d["dam_foals_sweet"] = np.zeros(n)
        # 父EI × 母賞金交互作用（99パーセンタイル正規化）
        if "sire_dam_interaction" in df.columns:
            inter = df["sire_dam_interaction"].fillna(0).values
            cap = np.percentile(inter, 99)
            d["sire_dam_inter_norm"] = (np.clip(inter, 0, cap) / cap) if cap > 0 else np.zeros(n)
        else:
            d["sire_dam_inter_norm"] = np.zeros(n)
        # 調教師 × 生産者コンボ（基準値2500超過分）
        if "trainer_breeder_combo" in df.columns:
            combo = df["trainer_breeder_combo"].fillna(2500).values
            d["trainer_breeder_excess"] = np.maximum(0, combo - 2500)
        else:
            d["trainer_breeder_excess"] = np.zeros(n)
        # 兄姉クラシック実績（時点制約済み）
        d["sibling_classic"] = df["sibling_classic"].fillna(0).values if "sibling_classic" in df.columns else np.zeros(n)
        # 種牡馬クラシック輩出数（時点制約済み）
        d["sire_classic_count"] = df["sire_classic_count"].fillna(0).values if "sire_classic_count" in df.columns else np.zeros(n)
        # 輸入繁殖牝馬フラグ
        d["imported_dam"] = df["imported_dam"].fillna(0).values if "imported_dam" in df.columns else np.zeros(n)
        # クラシック結果データ
        horse_ids = df["horse_id"].astype(str).values
        sex_vals = df["sex"].values if "sex" in df.columns else np.full(n, 0.5)
        derby_top5, oaks_top5 = _get_classic_ids(year)
        classic_top10 = derby_top5 | oaks_top5

        d["horse_ids"] = horse_ids
        d["sex_vals"] = sex_vals
        d["derby_idx"] = set(i for i, hid in enumerate(horse_ids) if hid in derby_top5)
        d["oaks_idx"] = set(i for i, hid in enumerate(horse_ids) if hid in oaks_top5)
        d["classic_idx"] = set(i for i, hid in enumerate(horse_ids) if hid in classic_top10)
        d["male_mask"] = (sex_vals == 1)
        d["female_mask"] = (sex_vals == 0)
        # male/female index mapping
        d["male_indices"] = np.where(d["male_mask"])[0]
        d["female_indices"] = np.where(d["female_mask"])[0]
        d["derby_in_males"] = set(i for i, gi in enumerate(d["male_indices"]) if horse_ids[gi] in derby_top5)
        d["oaks_in_females"] = set(i for i, gi in enumerate(d["female_indices"]) if horse_ids[gi] in oaks_top5)

        precomputed[year] = d
    return precomputed


def _compute_score_vec(d, params):
    """事前計算配列からスコアベクトルを計算する。"""
    score = (
        d["sex_centered"] * params[0]          # b_sex
        + d["sire_ei_norm"] * params[1]         # w_sire_ei
        + d["dam_prize_norm"] * params[2]       # w_dam_prize
        + d["bms_ei_norm"] * params[3]          # w_bms_ei
        + d["first_crop_val"] * params[4]       # w_first_crop
        + d["trainer_centered"] * params[5]     # w_trainer
        + d["owner_centered"] * params[6]       # w_owner
        + d["breeder_centered"] * params[7]     # w_breeder
        + d["early_born"] * params[8]           # b_early
        + d["dam_bms_gap"] * params[10]         # b_dam_bms_gap
        + d["sale_price_norm"] * params[11]     # b_sale_price
        + d["is_first_foal"] * (-params[12])    # b_foal_penalty
        + d["is_good_foal"] * params[13]        # b_foal_bonus
        - d["sire_old_excess"] * params[17]     # b_sire_old
    )
    if "parents_young" in d:
        score += d["parents_young"] * params[9]
    else:
        score += d["parents_young_half"] * (params[9] / 2)
    breed_val = np.clip(params[14] - d["dam_breed_age"], -params[16], params[15])
    score += d["dam_breed_notna"] * breed_val
    # 新特徴量
    score += d["dam_foals_sweet"] * params[18]         # b_dam_foals_sweet
    score += d["sire_dam_inter_norm"] * params[19]     # w_sire_dam_inter
    score += d["trainer_breeder_excess"] * params[20]  # w_trainer_breeder
    score += d["sibling_classic"] * params[21]         # b_sibling_classic
    score += d["sire_classic_count"] * params[22]      # w_sire_classic
    score += d["imported_dam"] * params[23]            # b_imported_dam
    return score


def _fast_cv_score(precomputed, params):
    """事前計算配列を使った高速CVスコア（全体TOP10 + 牡ダービー + 牝オークス）。"""
    total_score = 0.0
    n_years = len(precomputed)

    for d in precomputed.values():
        score = _compute_score_vec(d, params)
        n_horses = len(score)

        # 全体TOP10/TOP20/TOP30でクラシックTOP10ヒット
        top10_k = min(10, n_horses)
        pred_top10 = set(np.argpartition(-score, top10_k)[:top10_k])
        top10_match = len(pred_top10 & d["classic_idx"])

        top20_k = min(20, n_horses)
        pred_top20 = set(np.argpartition(-score, top20_k)[:top20_k])
        top20_match = len(pred_top20 & d["classic_idx"])

        top30_k = min(30, n_horses)
        pred_top30 = set(np.argpartition(-score, top30_k)[:top30_k])
        top30_match = len(pred_top30 & d["classic_idx"])

        # 牡馬TOP10/20内のダービーヒット
        m_d10 = m_d20 = 0
        male_idx = d["male_indices"]
        if len(male_idx) > 0 and d["derby_idx"]:
            male_scores = score[male_idx]
            mk10 = min(10, len(male_scores))
            m_top10 = set(np.argpartition(-male_scores, mk10)[:mk10])
            m_d10 = len(m_top10 & d["derby_in_males"])
            mk20 = min(20, len(male_scores))
            m_top20 = set(np.argpartition(-male_scores, mk20)[:mk20])
            m_d20 = len(m_top20 & d["derby_in_males"])

        # 牝馬TOP10/20内のオークスヒット
        f_o10 = f_o20 = 0
        female_idx = d["female_indices"]
        if len(female_idx) > 0 and d["oaks_idx"]:
            female_scores = score[female_idx]
            fk10 = min(10, len(female_scores))
            f_top10 = set(np.argpartition(-female_scores, fk10)[:fk10])
            f_o10 = len(f_top10 & d["oaks_in_females"])
            fk20 = min(20, len(female_scores))
            f_top20 = set(np.argpartition(-female_scores, fk20)[:fk20])
            f_o20 = len(f_top20 & d["oaks_in_females"])

        # ランク平滑化: クラシック馬のスコア合計（滑らかな勾配を提供）
        smooth_bonus = 0.0
        if d["classic_idx"]:
            classic_score_sum = sum(score[idx] for idx in d["classic_idx"])
            smooth_bonus = classic_score_sum * 0.01

        total_score += (
            top10_match * 100.0    # 全体TOP10ヒットが最重要
            + top20_match * 10.0   # 全体TOP20
            + top30_match * 3.0    # 全体TOP30
            + m_d10 * 80.0         # 牡馬TOP10内ダービーヒット
            + m_d20 * 8.0          # 牡馬TOP20内ダービーヒット
            + f_o10 * 80.0         # 牝馬TOP10内オークスヒット
            + f_o20 * 8.0          # 牝馬TOP20内オークスヒット
            + smooth_bonus         # ランク平滑化
        )

    return total_score / n_years


# パラメータ名 → 配列インデックスの対応
_PARAM_KEYS = [
    "b_sex", "w_sire_ei", "w_dam_prize", "w_bms_ei", "w_first_crop",
    "w_trainer", "w_owner", "w_breeder", "b_early", "b_parents_young",
    "b_dam_bms_gap", "b_sale_price", "b_foal_penalty", "b_foal_bonus",
    "dam_breed_base", "dam_breed_cap", "dam_breed_penalty",
    "b_sire_old",
    "b_dam_foals_sweet", "w_sire_dam_inter", "w_trainer_breeder",
    "b_sibling_classic", "w_sire_classic",
    "b_imported_dam",
]

def _dict_to_arr(params):
    return np.array([params[k] for k in _PARAM_KEYS])

def _arr_to_dict(arr):
    return {k: float(v) for k, v in zip(_PARAM_KEYS, arr)}


def _random_search_top10(all_data, current_params, best_score):
    """scipy Differential Evolution + ランダム局所探索（TOP10最大化）。"""
    import time
    from scipy.optimize import differential_evolution

    # 事前計算（1回だけ）
    print("\n  特徴量を事前計算中...")
    precomputed = _precompute_arrays(all_data)
    print("  完了")

    # パラメータの探索範囲（やや広め）
    bounds = [
        (0, 25),       # b_sex
        (0.0, 0.60),   # w_sire_ei
        (0.0, 0.40),   # w_dam_prize
        (0.0, 0.45),   # w_bms_ei
        (0.0, 30.0),   # w_first_crop
        (0.0, 0.60),   # w_trainer
        (0.0, 0.60),   # w_owner
        (0.0, 0.60),   # w_breeder
        (0, 20),       # b_early
        (0, 20),       # b_parents_young
        (0, 20),       # b_dam_bms_gap
        (0, 20),       # b_sale_price
        (0, 20),       # b_foal_penalty
        (0, 15),       # b_foal_bonus
        (1, 15),       # dam_breed_base
        (0, 10),       # dam_breed_cap
        (0, 10),       # dam_breed_penalty
        (0, 8),        # b_sire_old
        (0, 15),       # b_dam_foals_sweet
        (0, 20),       # w_sire_dam_inter
        (0, 0.05),     # w_trainer_breeder
        (0, 30),       # b_sibling_classic
        (0, 5),        # w_sire_classic
        (0, 20),       # b_imported_dam
    ]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo
    n_params = len(lo)

    current_arr = _dict_to_arr(current_params)
    best_arr = current_arr.copy()
    best_score_fast = _fast_cv_score(precomputed, current_arr)
    print(f"  現行スコア(高速版): {best_score_fast:.2f}")

    def _show_top10(arr, label=""):
        params = _arr_to_dict(arr)
        items = []
        total_hits = 0
        derby_hits = 0
        oaks_hits = 0
        for y in sorted(all_data.keys()):
            m = evaluate_single(all_data[y], params, birth_year=y)
            t10 = m.get("top10_total", 0)
            md10 = m.get("male_top10_derby", 0)
            fo10 = m.get("female_top10_oaks", 0)
            items.append(f"{y}:{t10}(D{md10}O{fo10})")
            total_hits += t10
            derby_hits += md10
            oaks_hits += fo10
        print(f"  {label} TOP10={total_hits}/80 牡D={derby_hits}/40 牝O={oaks_hits}/40 [{', '.join(items)}]")

    t_start = time.time()

    # ================================================================
    # Phase 1: Differential Evolution（5リスタート）
    # ================================================================
    n_restarts = 5
    print(f"\n{'='*60}")
    print(f"  Phase 1: Differential Evolution ({n_restarts}リスタート)")
    print(f"{'='*60}")

    def de_objective(x):
        return -_fast_cv_score(precomputed, np.asarray(x))

    de_results = []
    for i, seed in enumerate([42, 137, 314, 577, 2024]):
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=500,
            popsize=15,
            tol=1e-5,
            seed=seed,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False,
            init='latinhypercube',
        )
        score = -result.fun
        de_results.append((score, result.x.copy()))
        elapsed = time.time() - t_start
        print(f"  DE#{i} (seed={seed}): {score:.2f} ({elapsed:.0f}s, nfev={result.nfev})")
        _show_top10(result.x, f"  DE#{i}")

        if score > best_score_fast:
            best_score_fast = score
            best_arr = result.x.copy()

    print(f"\n  Phase 1 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
    _show_top10(best_arr, "Best")

    # ================================================================
    # Phase 2: DE結果の上位候補を局所探索（上位5 × 50k）
    # ================================================================
    de_results.sort(key=lambda x: -x[0])
    n_top = min(5, len(de_results))
    n_phase2 = 50000
    print(f"\n{'='*60}")
    print(f"  Phase 2: 上位{n_top}候補の局所探索 ({n_top} × {n_phase2:,} = {n_top*n_phase2:,}回)")
    print(f"{'='*60}")

    rng = np.random.default_rng(9999)
    for ci in range(n_top):
        cand_score, cand_arr = de_results[ci]
        local_best = cand_score
        local_arr = cand_arr.copy()

        for _ in range(n_phase2):
            delta = rng.uniform(-0.10, 0.10, n_params) * span
            p = np.clip(local_arr + delta, lo, hi)
            s = _fast_cv_score(precomputed, p)
            if s > local_best:
                local_best = s
                local_arr = p.copy()

        _show_top10(local_arr, f"Cand {ci}: score={local_best:.2f}")
        if local_best > best_score_fast:
            best_score_fast = local_best
            best_arr = local_arr.copy()

    print(f"\n  Phase 2 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
    _show_top10(best_arr, "Best")

    # ================================================================
    # Phase 3: 微調整（300k at ±5% → 200k at ±2%）
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Phase 3: 微調整")
    print(f"{'='*60}")

    for step_size, n_iter, label in [
        (0.05, 300000, "±5%"),
        (0.02, 200000, "±2%"),
    ]:
        improved = 0
        for _ in range(n_iter):
            delta = rng.uniform(-step_size, step_size, n_params) * span
            p = np.clip(best_arr + delta, lo, hi)
            s = _fast_cv_score(precomputed, p)
            if s > best_score_fast:
                best_score_fast = s
                best_arr = p.copy()
                improved += 1
        elapsed = time.time() - t_start
        print(f"  {label}: {best_score_fast:.2f} (改善{improved}回, {elapsed:.0f}s)")

    _show_top10(best_arr, "Final")

    # dict形式で返す（元のcv_scoreで検算）
    best_params = _arr_to_dict(best_arr)
    best_score = cv_score(all_data, best_params)
    print(f"\n  検算(元スコア関数): {best_score:.2f}")
    print(f"  総所要時間: {time.time()-t_start:.0f}秒")

    return best_params, best_score


def _staged_grid_search(all_data, best_params, best_score):
    """従来の段階的グリッドサーチ（balanced用）。"""
    # ===============================================================
    # Stage 1: 血統重み（父EI + 母父EI + 母馬賞金 + 初年度ボーナス）
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 1: 血統重み最適化")
    print(f"{'='*60}")
    blood_combos = list(itertools.product(
        [0.15, 0.20, 0.225, 0.25, 0.30, 0.35],   # w_sire_ei
        [0.05, 0.075, 0.10, 0.15, 0.20],           # w_dam_prize
        [0.10, 0.15, 0.20, 0.25, 0.30],            # w_bms_ei
        [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],            # w_first_crop
    ))
    print(f"  組み合わせ数: {len(blood_combos)}")

    for ws, wd, wb, wf in blood_combos:
        p = dict(best_params)
        p["w_sire_ei"] = ws
        p["w_dam_prize"] = wd
        p["w_bms_ei"] = wb
        p["w_first_crop"] = wf
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"w_sire_ei": ws, "w_dam_prize": wd,
                                "w_bms_ei": wb, "w_first_crop": wf})

    print(f"  最良: sire_ei={best_params['w_sire_ei']}, dam_prize={best_params['w_dam_prize']}, "
          f"bms_ei={best_params['w_bms_ei']}, first_crop={best_params['w_first_crop']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 2: ボーナスパラメータ
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 2: ボーナス最適化")
    print(f"{'='*60}")
    bonus_combos = list(itertools.product(
        [0, 3, 5, 8, 10, 12],     # b_early
        [0, 3, 5, 8, 10, 12],     # b_parents_young
        [0, 2, 5, 8],             # b_dam_bms_gap
        [0, 3, 5, 8, 10],         # b_sale_price
    ))
    print(f"  組み合わせ数: {len(bonus_combos)}")

    for be, bp, bg, bs in bonus_combos:
        p = dict(best_params)
        p["b_early"] = be
        p["b_parents_young"] = bp
        p["b_dam_bms_gap"] = bg
        p["b_sale_price"] = bs
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"b_early": be, "b_parents_young": bp,
                                "b_dam_bms_gap": bg, "b_sale_price": bs})

    print(f"  最良: early={best_params['b_early']}, parents_young={best_params['b_parents_young']}, "
          f"dam_bms_gap={best_params['b_dam_bms_gap']}, sale_price={best_params['b_sale_price']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 3: コネクション重み + 産駒番号
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 3: コネクション + 産駒番号")
    print(f"{'='*60}")
    conn_combos = list(itertools.product(
        [0, 2, 3, 5, 7],              # b_foal_penalty
        [0, 1, 2, 3, 5],              # b_foal_bonus
        [0.03, 0.05, 0.08, 0.12],     # w_trainer
        [0.03, 0.05, 0.08, 0.12],     # w_owner
        [0.05, 0.10, 0.15, 0.20],     # w_breeder
    ))
    print(f"  組み合わせ数: {len(conn_combos)}")

    for fp, fb, wt, wo, wb in conn_combos:
        p = dict(best_params)
        p["b_foal_penalty"] = fp
        p["b_foal_bonus"] = fb
        p["w_trainer"] = wt
        p["w_owner"] = wo
        p["w_breeder"] = wb
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"b_foal_penalty": fp, "b_foal_bonus": fb,
                                "w_trainer": wt, "w_owner": wo, "w_breeder": wb})

    print(f"  最良: foal_penalty={best_params['b_foal_penalty']}, foal_bonus={best_params['b_foal_bonus']}, "
          f"trainer={best_params['w_trainer']}, owner={best_params['w_owner']}, breeder={best_params['w_breeder']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 4: 繁殖入り年齢パラメータ
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 4: 繁殖入り年齢パラメータ")
    print(f"{'='*60}")
    breed_combos = list(itertools.product(
        [5, 6, 7, 8],     # dam_breed_base
        [2, 3, 4, 5],     # dam_breed_cap
        [1, 2, 3],        # dam_breed_penalty
    ))
    print(f"  組み合わせ数: {len(breed_combos)}")

    for base, cap, pen in breed_combos:
        p = dict(best_params)
        p["dam_breed_base"] = base
        p["dam_breed_cap"] = cap
        p["dam_breed_penalty"] = pen
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"dam_breed_base": base, "dam_breed_cap": cap,
                                "dam_breed_penalty": pen})

    print(f"  最良: base={best_params['dam_breed_base']}, cap={best_params['dam_breed_cap']}, "
          f"penalty={best_params['dam_breed_penalty']}")
    print(f"  CVスコア: {best_score:.2f}")

    # ===============================================================
    # Stage 5: 血統重み微調整
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  Stage 5: 血統重み微調整")
    print(f"{'='*60}")
    def _fine(val, step=0.025):
        return sorted(set([val + d for d in [-step, -step/2, 0, step/2, step]]))

    fine_combos = list(itertools.product(
        _fine(best_params["w_sire_ei"]),
        _fine(best_params["w_dam_prize"]),
        _fine(best_params["w_bms_ei"]),
        _fine(best_params["w_first_crop"], step=0.1),
    ))
    print(f"  組み合わせ数: {len(fine_combos)}")

    for ws, wd, wb, wf in fine_combos:
        if ws <= 0 or wd <= 0 or wb <= 0 or wf <= 0:
            continue
        p = dict(best_params)
        p["w_sire_ei"] = ws
        p["w_dam_prize"] = wd
        p["w_bms_ei"] = wb
        p["w_first_crop"] = wf
        s = cv_score(all_data, p)
        if s > best_score:
            best_score = s
            best_params.update({"w_sire_ei": ws, "w_dam_prize": wd,
                                "w_bms_ei": wb, "w_first_crop": wf})

    return best_params, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="グリッドサーチ（重み最適化）")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                        help="使用する年度リスト（生年）")
    parser.add_argument("--objective", choices=["classic", "top10"],
                        default="top10",
                        help="最適化目的: classic(バランス) / top10(ヒット数最大化)")
    args = parser.parse_args()

    best = grid_search(args.years, objective=args.objective)
