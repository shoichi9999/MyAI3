"""
グリッドサーチ — ヒューリスティックスコアの重み最適化。

ダービー（牡馬）またはオークス（牝馬）TOP5予測を最大化するようにパラメータを探索する。

使い方:
  python grid_search.py --years 2019
  python grid_search.py --race oaks          # オークス用最適化
  python grid_search.py --race derby         # ダービー用最適化（デフォルト）
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
    """パラメータ辞書でスコアを計算する（牡馬ダービー特化）。"""
    score = pd.Series(0.0, index=df.index)

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

    # 種牡馬産駒総賞金
    if "sire_progeny_prize" in df.columns:
        pp = np.log1p(df["sire_progeny_prize"].fillna(0))
        cap = pp.quantile(0.99)
        if cap > 0:
            score += (pp.clip(upper=cap) / cap) * 100 * params.get("w_sire_progeny_prize", 0)

    # 種牡馬産駒数
    if "sire_runners" in df.columns:
        sr = np.log1p(df["sire_runners"].fillna(0))
        cap = sr.quantile(0.99)
        if cap > 0:
            score += (sr.clip(upper=cap) / cap) * 100 * params.get("w_sire_runners", 0)

    # 母父産駒数
    if "bms_runners" in df.columns:
        br = np.log1p(df["bms_runners"].fillna(0))
        cap = br.quantile(0.99)
        if cap > 0:
            score += (br.clip(upper=cap) / cap) * 100 * params.get("w_bms_runners", 0)

    # 母馬産駒品質
    if "dam_progeny_quality" in df.columns:
        score += df["dam_progeny_quality"].fillna(0) * params.get("b_dam_progeny_quality", 0)

    # 種牡馬勝率
    if "sire_win_rate" in df.columns:
        wr = df["sire_win_rate"].fillna(0)
        score += wr * 100 * params.get("w_sire_win_rate", 0)

    # 母父勝率
    if "bms_win_rate" in df.columns:
        wr = df["bms_win_rate"].fillna(0)
        score += wr * 100 * params.get("w_bms_win_rate", 0)

    # 種牡馬ランクスコア
    if "sire_rank_score" in df.columns:
        rs = df["sire_rank_score"].fillna(0)
        cap = rs.max()
        if cap > 0:
            score += (rs / cap) * 100 * params.get("w_sire_rank", 0)

    # 母父ランクスコア
    if "bms_rank_score" in df.columns:
        rs = df["bms_rank_score"].fillna(0)
        cap = rs.max()
        if cap > 0:
            score += (rs / cap) * 100 * params.get("w_bms_rank", 0)

    # 母父産駒総賞金（対数 + 99パーセンタイル正規化）
    if "bms_progeny_prize" in df.columns:
        pp = np.log1p(df["bms_progeny_prize"].fillna(0))
        cap = pp.quantile(0.99)
        if cap > 0:
            score += (pp.clip(upper=cap) / cap) * 100 * params.get("w_bms_progeny_prize", 0)

    # 種牡馬クラシック率
    if "sire_classic_rate" in df.columns:
        score += df["sire_classic_rate"].fillna(0) * params.get("w_sire_classic_rate", 0)

    # 馬主 × 調教師コンボ
    if "owner_trainer_combo" in df.columns:
        combo = df["owner_trainer_combo"].fillna(2500)
        score += np.maximum(0, combo - 2500) * params.get("w_owner_trainer", 0)

    # 母父EI × 母賞金交互作用
    if "bms_dam_interaction" in df.columns:
        inter = df["bms_dam_interaction"].fillna(0)
        cap = inter.quantile(0.99)
        if cap > 0:
            score += (inter.clip(upper=cap) / cap) * params.get("w_bms_dam_inter", 0)

    # 種牡馬オークスクラシック率
    if "sire_oaks_rate" in df.columns:
        score += df["sire_oaks_rate"].fillna(0) * params.get("w_sire_oaks_rate", 0)

    # 母父クラシック輩出数
    if "bms_classic_count" in df.columns:
        score += df["bms_classic_count"].fillna(0) * params.get("w_bms_classic", 0)

    # 母馬高クラスフラグ
    if "dam_high_class" in df.columns:
        score += df["dam_high_class"].fillna(0) * params.get("b_dam_high_class", 0)

    # 種牡馬2歳EI（早熟性の直接指標）
    if "sire_2yo_ei" in df.columns:
        ei_2yo = df["sire_2yo_ei"].fillna(0)
        cap = ei_2yo.quantile(0.99)
        if cap > 0:
            score += (ei_2yo.clip(upper=cap) / cap) * 100 * params.get("w_sire_2yo_ei", 0)

    # 種牡馬の早熟性比率（2歳EI/全体EI）
    if "sire_precocity" in df.columns:
        prec = df["sire_precocity"].fillna(0)
        cap = prec.quantile(0.99)
        if cap > 0:
            score += (prec.clip(upper=cap) / cap) * 100 * params.get("w_sire_precocity", 0)

    # 種牡馬EIトレンド（上昇 = 加点）
    if "sire_ei_trend" in df.columns:
        trend = df["sire_ei_trend"].fillna(0)
        score += trend * params.get("w_sire_ei_trend", 0)

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


def _get_classic_ids(birth_year: int, race_type: str = "derby") -> set:
    """指定レースTOP5のhorse_idセットを返す。"""
    return set(_CLASSIC.get(race_type, {}).get(str(birth_year), []))


def _get_classic_winner_id(birth_year: int, race_type: str = "derby") -> str | None:
    """指定レース1着馬のhorse_idを返す。"""
    classic_list = _CLASSIC.get(race_type, {}).get(str(birth_year), [])
    return classic_list[0] if classic_list else None


# 後方互換エイリアス
def _get_derby_ids(birth_year: int) -> set:
    return _get_classic_ids(birth_year, "derby")

def _get_derby_winner_id(birth_year: int) -> str | None:
    return _get_classic_winner_id(birth_year, "derby")


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_single(df: pd.DataFrame, params: dict, birth_year: int = None,
                    race_type: str = "derby") -> dict:
    """1年度のデータで指定レースの予測を評価する。"""
    scores = parameterized_score(df, params).values
    horse_ids = df["horse_id"].astype(str).values

    if birth_year is None:
        birth_year = df.get("_birth_year", 2020)

    classic_top5 = _get_classic_ids(birth_year, race_type)
    classic_winner_id = _get_classic_winner_id(birth_year, race_type)

    results = {}

    # 予測TOP N にTOP5が何頭入るか
    for n in [5, 10, 15, 20, 30]:
        pred_idx = np.argsort(-scores)[:n]
        pred_ids = set(horse_ids[pred_idx])
        results[f"top{n}_{race_type}"] = len(pred_ids & classic_top5)

    # 1着馬の予測順位
    if classic_winner_id:
        all_ranks = np.argsort(np.argsort(-scores)) + 1
        idx = np.where(horse_ids == classic_winner_id)[0]
        if len(idx) > 0:
            results["winner_rank"] = int(all_ranks[idx[0]])
        else:
            results["winner_rank"] = len(horse_ids)
    else:
        results["winner_rank"] = len(horse_ids)

    return results


def composite_score(metrics: dict, objective: str = "classic",
                    race_type: str = "derby") -> float:
    """複合スコア（1着馬とTOP5ヒットのバランス改善版）。"""
    d5 = metrics.get(f"top5_{race_type}", 0)
    d10 = metrics.get(f"top10_{race_type}", 0)
    winner_rank = metrics.get("winner_rank", 9999)

    # 1着馬の段階ボーナス（バランス改善: 旧1200→500）
    winner_bonus = 0.0
    if winner_rank <= 3:
        winner_bonus = 500.0
    elif winner_rank <= 5:
        winner_bonus = 400.0
    elif winner_rank <= 10:
        winner_bonus = 200.0
    elif winner_rank <= 20:
        winner_bonus = 80.0
    elif winner_rank <= 30:
        winner_bonus = 30.0
    elif winner_rank <= 50:
        winner_bonus = 10.0

    return (
        winner_bonus           # 1着馬の順位
        + d5 * 100.0           # TOP5ヒット（40→100: 重要度UP）
        + d10 * 30.0           # TOP10ヒット（新規追加）
    )


def cv_score(all_data: dict, params: dict, race_type: str = "derby") -> float:
    """全年度の composite_score 平均を返す。"""
    scores = []
    for year, df in all_data.items():
        m = evaluate_single(df, params, birth_year=year, race_type=race_type)
        scores.append(composite_score(m, race_type=race_type))
    return np.mean(scores)


# ------------------------------------------------------------------
# データ読み込み
# ------------------------------------------------------------------

def load_year(year: int, race_type: str = "derby") -> pd.DataFrame | None:
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        return None
    # レース結果がない年はスキップ
    classic_ids = _get_classic_ids(year, race_type)
    if not classic_ids:
        return None
    horses = pd.read_csv(csv_path)
    sex_filter = "牡" if race_type == "derby" else "牝"
    features = build_feature_matrix(horses, birth_year=year, sex_filter=sex_filter)
    return features


# ------------------------------------------------------------------
# グリッドサーチ
# ------------------------------------------------------------------

def grid_search(years, objective="balanced", race_type="derby"):
    global _OBJECTIVE, _RACE_TYPE
    _OBJECTIVE = objective
    _RACE_TYPE = race_type
    race_label = "ダービー（牡馬）" if race_type == "derby" else "オークス（牝馬）"
    print(f"=== データ読み込み === (目的関数: {objective}, レース: {race_label})")
    all_data = {}
    for y in years:
        df = load_year(y, race_type=race_type)
        if df is not None:
            all_data[y] = df
            print(f"  {y}年: {len(df)}頭")
        else:
            print(f"  {y}年: データなし（スキップ）")

    if not all_data:
        print("[ERROR] データが見つかりません")
        return

    print(f"\n  利用年度: {sorted(all_data.keys())} ({len(all_data)}年分)")

    # 現行パラメータ（レースタイプに応じた重みファイルから読み込み）
    from src.model import _load_weights
    _w = _load_weights(race_type)
    current_params = {
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
        "w_sire_win_rate": _w.get("w_sire_win_rate", 0.0),
        "w_bms_win_rate": _w.get("w_bms_win_rate", 0.0),
        "w_sire_rank": _w.get("w_sire_rank", 0.0),
        "w_sire_progeny_prize": _w.get("w_sire_progeny_prize", 0.0),
        "w_sire_runners": _w.get("w_sire_runners", 0.0),
        "w_bms_runners": _w.get("w_bms_runners", 0.0),
        "b_dam_progeny_quality": _w.get("b_dam_progeny_quality", 0.0),
        "w_bms_rank": _w.get("w_bms_rank", 0.0),
        "w_bms_progeny_prize": _w.get("w_bms_progeny_prize", 0.0),
        "w_sire_classic_rate": _w.get("w_sire_classic_rate", 0.0),
        "w_owner_trainer": _w.get("w_owner_trainer", 0.0),
        "w_bms_dam_inter": _w.get("w_bms_dam_inter", 0.0),
        "w_sire_2yo_ei": _w.get("w_sire_2yo_ei", 0.0),
        "w_sire_precocity": _w.get("w_sire_precocity", 0.0),
        "w_sire_ei_trend": _w.get("w_sire_ei_trend", 0.0),
        "w_sire_oaks_rate": _w.get("w_sire_oaks_rate", 0.0),
        "w_bms_classic": _w.get("w_bms_classic", 0.0),
        "b_dam_high_class": _w.get("b_dam_high_class", 0.0),
    }

    current_cv = cv_score(all_data, current_params, race_type=race_type)
    print(f"\n  現行パラメータのCVスコア: {current_cv:.2f}")
    winner_in_top10 = 0
    for y, df in sorted(all_data.items()):
        m = evaluate_single(df, current_params, birth_year=y, race_type=race_type)
        d10 = m.get(f"top10_{race_type}", 0)
        wr = m.get("winner_rank", "?")
        hit = "✓" if isinstance(wr, int) and wr <= 10 else "✗"
        if isinstance(wr, int) and wr <= 10:
            winner_in_top10 += 1
        print(f"    {y}年: TOP10={d10}/5  1着={wr}位 {hit}")
    print(f"  1着TOP10入り: {winner_in_top10}/{len(all_data)}年")

    best_score = current_cv
    best_params = dict(current_params)

    if objective == "top10":
        # TOP10最適化: 全パラメータ同時ランダム探索（局所最適回避）
        best_params, best_score = _random_search_top10(all_data, current_params, best_score, race_type=race_type)
    else:
        # balanced: 従来の段階的グリッドサーチ
        best_params, best_score = _staged_grid_search(all_data, dict(current_params), best_score, race_type=race_type)

    # ------------------------------------------------------------------
    # 最終結果
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  最適パラメータ ({race_label})")
    print("=" * 70)
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")
    print(f"\n  CVスコア: {best_score:.2f} (現行: {current_cv:.2f}, 差: {best_score - current_cv:+.2f})")

    # 重みファイルに書き戻し（レースタイプに応じて保存先を切替）
    import json as _json
    if race_type == "oaks":
        weights_path = "data/config/weights_oaks.json"
    else:
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
    print(f"\n--- 年度別パフォーマンス比較（1着TOP10入り） ---")
    print(f"{'年':>6} | {'指標':>12} | {'現行':>8} | {'最適化':>8} | {'差':>6}")
    print("-" * 60)
    old_w_hits = new_w_hits = 0
    for y in sorted(all_data.keys()):
        m_old = evaluate_single(all_data[y], current_params, birth_year=y, race_type=race_type)
        m_new = evaluate_single(all_data[y], best_params, birth_year=y, race_type=race_type)
        # ダービー1着順位
        wr_old = m_old.get("winner_rank", "?")
        wr_new = m_new.get("winner_rank", "?")
        hit_old = "✓" if isinstance(wr_old, int) and wr_old <= 10 else "✗"
        hit_new = "✓" if isinstance(wr_new, int) and wr_new <= 10 else "✗"
        if isinstance(wr_old, int) and wr_old <= 10: old_w_hits += 1
        if isinstance(wr_new, int) and wr_new <= 10: new_w_hits += 1
        print(f"  {y} | {'1着順位':>12} | {wr_old:>5}位{hit_old} | {wr_new:>5}位{hit_new} | {(wr_new-wr_old) if isinstance(wr_old, int) and isinstance(wr_new, int) else '':>+5}")
        for n in [10, 20, 30]:
            old_val = m_old.get(f"top{n}_{race_type}", 0)
            new_val = m_new.get(f"top{n}_{race_type}", 0)
            print(f"  {y} | {'TOP'+str(n):>12} | {old_val:>4}/5   | {new_val:>4}/5   | {new_val-old_val:>+5}")
        print("-" * 60)
    print(f"  1着TOP10入り: 現行={old_w_hits}/8  最適化={new_w_hits}/8")

    return best_params


def _precompute_arrays(all_data, race_type="derby"):
    """DataFrameから高速評価用のnumpy配列を事前計算する。"""
    precomputed = {}
    for year, df in all_data.items():
        n = len(df)
        d = {}
        # (性別関連削除 — 牡馬ダービー特化のため全馬が牡馬)
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
        # 種牡馬産駒総賞金（対数正規化）
        if "sire_progeny_prize" in df.columns:
            pp = np.log1p(df["sire_progeny_prize"].fillna(0).values)
            cap = np.percentile(pp, 99)
            d["sire_progeny_prize_norm"] = (np.clip(pp, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_progeny_prize_norm"] = np.zeros(n)
        # 種牡馬産駒数（対数正規化）
        if "sire_runners" in df.columns:
            sr = np.log1p(df["sire_runners"].fillna(0).values)
            cap = np.percentile(sr, 99)
            d["sire_runners_norm"] = (np.clip(sr, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_runners_norm"] = np.zeros(n)
        # 母父産駒数（対数正規化）
        if "bms_runners" in df.columns:
            br = np.log1p(df["bms_runners"].fillna(0).values)
            cap = np.percentile(br, 99)
            d["bms_runners_norm"] = (np.clip(br, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["bms_runners_norm"] = np.zeros(n)
        # 母馬産駒品質
        d["dam_progeny_quality"] = df["dam_progeny_quality"].fillna(0).values if "dam_progeny_quality" in df.columns else np.zeros(n)
        # 種牡馬勝率（0-1値を*100して正規化）
        d["sire_win_rate_100"] = (df["sire_win_rate"].fillna(0) * 100).values if "sire_win_rate" in df.columns else np.zeros(n)
        # 母父勝率
        d["bms_win_rate_100"] = (df["bms_win_rate"].fillna(0) * 100).values if "bms_win_rate" in df.columns else np.zeros(n)
        # 種牡馬ランクスコア（正規化済み）
        if "sire_rank_score" in df.columns:
            rs = df["sire_rank_score"].fillna(0).values
            cap = rs.max()
            d["sire_rank_norm"] = (rs / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_rank_norm"] = np.zeros(n)
        # 母父ランクスコア（正規化済み）
        if "bms_rank_score" in df.columns:
            rs = df["bms_rank_score"].fillna(0).values
            cap = rs.max()
            d["bms_rank_norm"] = (rs / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["bms_rank_norm"] = np.zeros(n)
        # 母父産駒総賞金（対数正規化）
        if "bms_progeny_prize" in df.columns:
            pp = np.log1p(df["bms_progeny_prize"].fillna(0).values)
            cap = np.percentile(pp, 99)
            d["bms_progeny_prize_norm"] = (np.clip(pp, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["bms_progeny_prize_norm"] = np.zeros(n)
        # 種牡馬クラシック率
        d["sire_classic_rate"] = df["sire_classic_rate"].fillna(0).values if "sire_classic_rate" in df.columns else np.zeros(n)
        # 馬主 × 調教師コンボ（基準値2500超過分）
        if "owner_trainer_combo" in df.columns:
            combo = df["owner_trainer_combo"].fillna(2500).values
            d["owner_trainer_excess"] = np.maximum(0, combo - 2500)
        else:
            d["owner_trainer_excess"] = np.zeros(n)
        # 母父EI × 母賞金交互作用（99パーセンタイル正規化）
        if "bms_dam_interaction" in df.columns:
            inter = df["bms_dam_interaction"].fillna(0).values
            cap = np.percentile(inter, 99)
            d["bms_dam_inter_norm"] = (np.clip(inter, 0, cap) / cap) if cap > 0 else np.zeros(n)
        else:
            d["bms_dam_inter_norm"] = np.zeros(n)
        # 種牡馬2歳EI（正規化済み）
        if "sire_2yo_ei" in df.columns:
            ei_2yo = df["sire_2yo_ei"].fillna(0).values
            cap = np.percentile(ei_2yo, 99)
            d["sire_2yo_ei_norm"] = (np.clip(ei_2yo, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_2yo_ei_norm"] = np.zeros(n)
        # 種牡馬早熟性比率（正規化済み）
        if "sire_precocity" in df.columns:
            prec = df["sire_precocity"].fillna(0).values
            cap = np.percentile(prec, 99)
            d["sire_precocity_norm"] = (np.clip(prec, 0, cap) / cap * 100) if cap > 0 else np.zeros(n)
        else:
            d["sire_precocity_norm"] = np.zeros(n)
        # 種牡馬EIトレンド（生値を使用 — 正負の方向性が重要）
        d["sire_ei_trend"] = df["sire_ei_trend"].fillna(0).values if "sire_ei_trend" in df.columns else np.zeros(n)
        # 種牡馬オークスクラシック率
        d["sire_oaks_rate"] = df["sire_oaks_rate"].fillna(0).values if "sire_oaks_rate" in df.columns else np.zeros(n)
        # 母父クラシック輩出数
        d["bms_classic_count"] = df["bms_classic_count"].fillna(0).values if "bms_classic_count" in df.columns else np.zeros(n)
        # 母馬高クラスフラグ
        d["dam_high_class"] = df["dam_high_class"].fillna(0).values if "dam_high_class" in df.columns else np.zeros(n)
        # クラシック結果データ（レースタイプに応じて切替）
        horse_ids = df["horse_id"].astype(str).values
        classic_top5 = _get_classic_ids(year, race_type)

        d["horse_ids"] = horse_ids
        d["derby_idx"] = set(i for i, hid in enumerate(horse_ids) if hid in classic_top5)

        # 1着馬のインデックス（連続順位ボーナス用）
        classic_winner_id = _get_classic_winner_id(year, race_type)
        d["derby_winner_idx"] = None
        if classic_winner_id:
            idxs = np.where(horse_ids == classic_winner_id)[0]
            if len(idxs) > 0:
                d["derby_winner_idx"] = idxs[0]

        precomputed[year] = d
    return precomputed


def _compute_score_vec(d, params):
    """事前計算配列からスコアベクトルを計算する（牡馬ダービー特化、b_sex削除）。"""
    score = (
        d["sire_ei_norm"] * params[0]           # w_sire_ei
        + d["dam_prize_norm"] * params[1]       # w_dam_prize
        + d["bms_ei_norm"] * params[2]          # w_bms_ei
        + d["first_crop_val"] * params[3]       # w_first_crop
        + d["trainer_centered"] * params[4]     # w_trainer
        + d["owner_centered"] * params[5]       # w_owner
        + d["breeder_centered"] * params[6]     # w_breeder
        + d["early_born"] * params[7]           # b_early
        + d["dam_bms_gap"] * params[9]          # b_dam_bms_gap
        + d["sale_price_norm"] * params[10]     # b_sale_price
        + d["is_first_foal"] * (-params[11])    # b_foal_penalty
        + d["is_good_foal"] * params[12]        # b_foal_bonus
        - d["sire_old_excess"] * params[16]     # b_sire_old
    )
    if "parents_young" in d:
        score += d["parents_young"] * params[8]
    else:
        score += d["parents_young_half"] * (params[8] / 2)
    breed_val = np.clip(params[13] - d["dam_breed_age"], -params[15], params[14])
    score += d["dam_breed_notna"] * breed_val
    # 新特徴量
    score += d["dam_foals_sweet"] * params[17]         # b_dam_foals_sweet
    score += d["sire_dam_inter_norm"] * params[18]     # w_sire_dam_inter
    score += d["trainer_breeder_excess"] * params[19]  # w_trainer_breeder
    score += d["sibling_classic"] * params[20]         # b_sibling_classic
    score += d["sire_classic_count"] * params[21]      # w_sire_classic
    score += d["imported_dam"] * params[22]            # b_imported_dam
    score += d["sire_win_rate_100"] * params[23]       # w_sire_win_rate
    score += d["bms_win_rate_100"] * params[24]        # w_bms_win_rate
    score += d["sire_rank_norm"] * params[25]          # w_sire_rank
    score += d["sire_progeny_prize_norm"] * params[26] # w_sire_progeny_prize
    score += d["sire_runners_norm"] * params[27]       # w_sire_runners
    score += d["bms_runners_norm"] * params[28]        # w_bms_runners
    score += d["dam_progeny_quality"] * params[29]     # b_dam_progeny_quality
    score += d["bms_rank_norm"] * params[30]           # w_bms_rank
    score += d["bms_progeny_prize_norm"] * params[31]  # w_bms_progeny_prize
    score += d["sire_classic_rate"] * params[32]       # w_sire_classic_rate
    score += d["owner_trainer_excess"] * params[33]    # w_owner_trainer
    score += d["bms_dam_inter_norm"] * params[34]      # w_bms_dam_inter
    score += d["sire_2yo_ei_norm"] * params[35]        # w_sire_2yo_ei
    score += d["sire_precocity_norm"] * params[36]     # w_sire_precocity
    score += d["sire_ei_trend"] * params[37]           # w_sire_ei_trend
    score += d["sire_oaks_rate"] * params[38]          # w_sire_oaks_rate
    score += d["bms_classic_count"] * params[39]       # w_bms_classic
    score += d["dam_high_class"] * params[40]          # b_dam_high_class
    return score


def _fast_cv_score(precomputed, params):
    """事前計算配列を使った高速CVスコア（過学習抑制版）。

    改善点:
    - 1着馬ボーナスとTOP5/TOP10ヒットのバランス改善
    - 滑らかな順位ボーナス（シグモイド的）で最適化しやすく
    - 年度間安定性ペナルティ（分散が大きい場合に減点）
    - L2正則化で極端なパラメータ値を抑制
    """
    year_scores = []
    n_years = len(precomputed)

    for d in precomputed.values():
        score = _compute_score_vec(d, params)
        n_horses = len(score)

        # TOP5/TOP10でクラシックTOP5ヒット
        d5 = 0
        d10 = 0
        if d["derby_idx"]:
            k5 = min(5, n_horses)
            pred_top5 = set(np.argpartition(-score, k5)[:k5])
            d5 = len(pred_top5 & d["derby_idx"])
            k10 = min(10, n_horses)
            pred_top10 = set(np.argpartition(-score, k10)[:k10])
            d10 = len(pred_top10 & d["derby_idx"])

        # ランク計算
        ranks = np.argsort(np.argsort(-score)) + 1  # 1-indexed

        # ===== 1着馬の順位（滑らかなボーナス） =====
        winner_bonus = 0.0
        widx = d["derby_winner_idx"]
        if widx is not None:
            rank_val = int(ranks[widx])
            pctl = 1.0 - rank_val / n_horses
            # 連続ボーナス（パーセンタイル）
            winner_bonus += pctl * 80.0
            # 段階ボーナス（TOP5重視だがバランス改善）
            if rank_val <= 3:
                winner_bonus += 500.0
            elif rank_val <= 5:
                winner_bonus += 400.0
            elif rank_val <= 10:
                winner_bonus += 200.0
            elif rank_val <= 20:
                winner_bonus += 80.0
            elif rank_val <= 30:
                winner_bonus += 30.0
            elif rank_val <= 50:
                winner_bonus += 10.0

        # クラシックTOP5全体のランク（重要度UP）
        smooth_bonus = 0.0
        if d["derby_idx"]:
            for idx in d["derby_idx"]:
                pctl = 1.0 - ranks[idx] / n_horses
                smooth_bonus += pctl * 20.0  # 3.0→20.0: 全体順位の重要度UP

        year_score = (
            winner_bonus
            + d5 * 100.0           # TOP5ヒット（40→100: 重要度UP）
            + d10 * 30.0           # TOP10ヒット（新規追加）
            + smooth_bonus
        )
        year_scores.append(year_score)

    # 年度平均
    mean_score = np.mean(year_scores)

    # 年度間安定性ペナルティ（標準偏差が大きいほど減点）
    if n_years > 1:
        std_penalty = np.std(year_scores) * 0.15
        mean_score -= std_penalty

    # L2正則化（極端なパラメータ値を抑制）
    if _BOUNDS_HI is not None:
        param_arr = np.asarray(params)
        # boundsのスパンで正規化してからL2計算
        l2_penalty = np.sum((param_arr / (_BOUNDS_HI + 1e-8)) ** 2) * 0.3
        mean_score -= l2_penalty

    return mean_score

# 探索範囲の上限（L2正則化用にグローバルで保持）
_BOUNDS_HI = None


# パラメータ名 → 配列インデックスの対応
_PARAM_KEYS = [
    "w_sire_ei", "w_dam_prize", "w_bms_ei", "w_first_crop",
    "w_trainer", "w_owner", "w_breeder", "b_early", "b_parents_young",
    "b_dam_bms_gap", "b_sale_price", "b_foal_penalty", "b_foal_bonus",
    "dam_breed_base", "dam_breed_cap", "dam_breed_penalty",
    "b_sire_old",
    "b_dam_foals_sweet", "w_sire_dam_inter", "w_trainer_breeder",
    "b_sibling_classic", "w_sire_classic",
    "b_imported_dam",
    "w_sire_win_rate", "w_bms_win_rate", "w_sire_rank",
    "w_sire_progeny_prize", "w_sire_runners", "w_bms_runners",
    "b_dam_progeny_quality",
    "w_bms_rank", "w_bms_progeny_prize", "w_sire_classic_rate",
    "w_owner_trainer", "w_bms_dam_inter",
    "w_sire_2yo_ei", "w_sire_precocity", "w_sire_ei_trend",
    "w_sire_oaks_rate", "w_bms_classic", "b_dam_high_class",
]

def _dict_to_arr(params):
    return np.array([params[k] for k in _PARAM_KEYS])

def _arr_to_dict(arr):
    return {k: float(v) for k, v in zip(_PARAM_KEYS, arr)}


def _random_search_top10(all_data, current_params, best_score, race_type="derby"):
    """scipy Differential Evolution + ランダム局所探索（TOP10最大化）。"""
    import time
    from scipy.optimize import differential_evolution

    # 事前計算（1回だけ）
    print("\n  特徴量を事前計算中...")
    precomputed = _precompute_arrays(all_data, race_type=race_type)
    print("  完了")

    # パラメータの探索範囲（現在の最適値を包含 + マージン）
    bounds = [
        (0.0, 1.5),    # w_sire_ei
        (0.0, 0.50),   # w_dam_prize
        (0.0, 0.50),   # w_bms_ei
        (0.0, 150.0),  # w_first_crop (現在値99.9を包含)
        (0.0, 3.0),    # w_trainer (現在値2.05を包含)
        (0.0, 1.5),    # w_owner
        (0.0, 1.5),    # w_breeder
        (0, 50),       # b_early (現在値38.3を包含)
        (0, 30),       # b_parents_young
        (0, 25),       # b_dam_bms_gap
        (0, 40),       # b_sale_price (現在値30.0を包含)
        (0, 25),       # b_foal_penalty (現在値19.9を包含)
        (0, 20),       # b_foal_bonus
        (1, 15),       # dam_breed_base
        (0, 20),       # dam_breed_cap (現在値14.1を包含)
        (0, 20),       # dam_breed_penalty
        (0, 10),       # b_sire_old
        (0, 30),       # b_dam_foals_sweet (現在値21.1を包含)
        (0, 25),       # w_sire_dam_inter
        (0, 0.20),     # w_trainer_breeder
        (0, 50),       # b_sibling_classic (現在値27.5を包含)
        (0, 8),        # w_sire_classic
        (0, 50),       # b_imported_dam (現在値33.8を包含)
        (0.0, 0.60),   # w_sire_win_rate
        (0.0, 0.60),   # w_bms_win_rate
        (0.0, 0.60),   # w_sire_rank
        (0.0, 0.50),   # w_sire_progeny_prize
        (0.0, 0.50),   # w_sire_runners
        (0.0, 0.50),   # w_bms_runners
        (0, 100),      # b_dam_progeny_quality
        (0.0, 0.60),   # w_bms_rank
        (0.0, 0.50),   # w_bms_progeny_prize
        (0.0, 20.0),   # w_sire_classic_rate
        (0.0, 0.20),   # w_owner_trainer
        (0.0, 30.0),   # w_bms_dam_inter (現在値21.4を包含)
        (0.0, 0.50),   # w_sire_2yo_ei
        (0.0, 0.50),   # w_sire_precocity
        (0.0, 30.0),   # w_sire_ei_trend
        (0.0, 20.0),   # w_sire_oaks_rate
        (0.0, 15.0),   # w_bms_classic
        (0.0, 50.0),   # b_dam_high_class
    ]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo
    n_params = len(lo)

    # L2正則化用のグローバル上限を設定
    global _BOUNDS_HI
    _BOUNDS_HI = hi

    current_arr = _dict_to_arr(current_params)
    best_arr = current_arr.copy()
    best_score_fast = _fast_cv_score(precomputed, current_arr)
    print(f"  現行スコア(高速版): {best_score_fast:.2f}")

    def _show_top10(arr, label=""):
        params = _arr_to_dict(arr)
        items = []
        total_d10 = 0
        winner_hits = 0
        for y in sorted(all_data.keys()):
            m = evaluate_single(all_data[y], params, birth_year=y, race_type=race_type)
            d10 = m.get(f"top10_{race_type}", 0)
            wr = m.get("winner_rank", "?")
            hit = "✓" if isinstance(wr, int) and wr <= 10 else ""
            if isinstance(wr, int) and wr <= 10:
                winner_hits += 1
            items.append(f"{y}:1着={wr}位{hit}")
            total_d10 += d10
        print(f"  {label} 1着TOP10={winner_hits}/8 T10={total_d10}/40 [{', '.join(items)}]")

    t_start = time.time()

    # ================================================================
    # Phase 1: Differential Evolution（7リスタート）
    # 注意: scipy DEのpopsizeは実人口 = popsize × n_params なので控えめに
    # ================================================================
    n_restarts = 7
    print(f"\n{'='*60}")
    print(f"  Phase 1: Differential Evolution ({n_restarts}リスタート)")
    print(f"{'='*60}")

    def de_objective(x):
        return -_fast_cv_score(precomputed, np.asarray(x))

    de_results = []
    de_seeds = [42, 137, 314, 577, 2024, 7777, 12345]
    for i, seed in enumerate(de_seeds[:n_restarts]):
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=400,
            popsize=8,             # 実人口 = 8 × n_params
            tol=1e-5,
            seed=seed,
            mutation=(0.5, 1.5),
            recombination=0.8,
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
    # Phase 2: CMA-ES（共分散行列適応進化戦略）
    # パラメータ間の相関を学習し、DEよりも効率的に探索
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Phase 2: CMA-ES (3リスタート)")
    print(f"{'='*60}")

    try:
        import cma

        # DE上位候補 + 現行パラメータを初期点として使用
        de_results.sort(key=lambda x: -x[0])
        cma_starts = [(best_score_fast, best_arr.copy())]
        for s, a in de_results[:2]:
            cma_starts.append((s, a.copy()))

        for ci, (start_score, start_arr) in enumerate(cma_starts):
            # 初期標準偏差: 探索範囲の15%
            sigma0 = 0.15 * np.mean(span)
            opts = cma.CMAOptions()
            opts.set('bounds', [lo.tolist(), hi.tolist()])
            opts.set('maxfevals', 15000)
            opts.set('verbose', -1)
            opts.set('seed', 42 + ci * 100)
            opts.set('tolfun', 1e-6)

            es = cma.CMAEvolutionStrategy(start_arr.tolist(), sigma0, opts)
            while not es.stop():
                solutions = es.ask()
                fitnesses = [-_fast_cv_score(precomputed, np.asarray(x)) for x in solutions]
                es.tell(solutions, fitnesses)

            cma_score = -es.result.fbest
            cma_arr = np.asarray(es.result.xbest)
            elapsed = time.time() - t_start
            print(f"  CMA#{ci}: {cma_score:.2f} (nfev={es.result.evaluations}, {elapsed:.0f}s)")
            _show_top10(cma_arr, f"CMA#{ci}")

            if cma_score > best_score_fast:
                best_score_fast = cma_score
                best_arr = cma_arr.copy()

        print(f"\n  Phase 2 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
        _show_top10(best_arr, "Best")

    except ImportError:
        print("  cma未インストール — Phase 2スキップ (pip install cma)")

    # ================================================================
    # Phase 3: DE上位候補の局所探索（上位5 × 50k）
    # ================================================================
    de_results.sort(key=lambda x: -x[0])
    n_top = min(5, len(de_results))
    n_phase3 = 50000
    print(f"\n{'='*60}")
    print(f"  Phase 3: 上位{n_top}候補の局所探索 ({n_top} × {n_phase3:,} = {n_top*n_phase3:,}回)")
    print(f"{'='*60}")

    rng = np.random.default_rng(9999)
    for ci in range(n_top):
        cand_score, cand_arr = de_results[ci]
        local_best = cand_score
        local_arr = cand_arr.copy()

        for _ in range(n_phase3):
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

    print(f"\n  Phase 3 完了: best={best_score_fast:.2f} ({time.time()-t_start:.0f}s)")
    _show_top10(best_arr, "Best")

    # ================================================================
    # Phase 4: 微調整（300k at ±5% → 200k at ±2% → 100k at ±1%）
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Phase 4: 微調整")
    print(f"{'='*60}")

    for step_size, n_iter, label in [
        (0.05, 300000, "±5%"),
        (0.02, 200000, "±2%"),
        (0.01, 100000, "±1%"),
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
    best_score = cv_score(all_data, best_params, race_type=race_type)
    print(f"\n  検算(元スコア関数): {best_score:.2f}")
    print(f"  総所要時間: {time.time()-t_start:.0f}秒")

    return best_params, best_score


def _staged_grid_search(all_data, best_params, best_score, race_type="derby"):
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
        s = cv_score(all_data, p, race_type=race_type)
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
        s = cv_score(all_data, p, race_type=race_type)
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
        s = cv_score(all_data, p, race_type=race_type)
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
        s = cv_score(all_data, p, race_type=race_type)
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
        s = cv_score(all_data, p, race_type=race_type)
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
    parser.add_argument("--race", choices=["derby", "oaks"],
                        default="derby",
                        help="対象レース: derby(ダービー・牡馬) / oaks(オークス・牝馬)")
    args = parser.parse_args()

    best = grid_search(args.years, objective=args.objective, race_type=args.race)
