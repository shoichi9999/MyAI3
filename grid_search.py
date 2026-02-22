"""
グリッドサーチ — ヒューリスティックスコアの重み最適化。

使い方:
  python grid_search.py --years 2019
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features import build_feature_matrix


# ------------------------------------------------------------------
# パラメータ化されたスコア関数
# ------------------------------------------------------------------

def parameterized_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """パラメータ辞書でスコアを計算する。"""
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

    # 初年度種牡馬ボーナス
    if "sire_prize" in df.columns and "sire_ei" in df.columns:
        is_first_crop = df["sire_ei"].fillna(0) == 0
        sire_prize_log = np.log1p(df["sire_prize"].fillna(0))
        score += is_first_crop * sire_prize_log * params["w_first_crop"]

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

    return score


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_single(df: pd.DataFrame, params: dict) -> dict:
    """1年度のデータで評価する。"""
    scores = parameterized_score(df, params).values
    y_true = df["prize_num"].values

    corr, _ = spearmanr(scores, y_true)
    if np.isnan(corr):
        corr = 0.0

    results = {}
    for n in [10, 30, 50, 100]:
        pred_idx = set(np.argsort(-scores)[:n])
        actual_idx = set(np.argsort(-y_true)[:n])
        results[f"top{n}"] = len(pred_idx & actual_idx)

    overall_avg = y_true.mean()
    pred_top30_avg = y_true[np.argsort(-scores)[:30]].mean()
    ratio = pred_top30_avg / overall_avg if overall_avg > 0 else 0

    return {
        "spearman": corr,
        "top10": results["top10"],
        "top30": results["top30"],
        "top50": results["top50"],
        "top100": results["top100"],
        "prize_ratio": ratio,
    }


def composite_score(metrics: dict) -> float:
    """POG実用性重視の複合スコア。"""
    return (
        metrics["top30"] * 2.0
        + metrics["top10"] * 3.0
        + metrics["top100"] * 0.5
        + metrics["prize_ratio"] * 0.3
        + metrics["spearman"] * 10
    )


def cv_score(all_data: dict, params: dict) -> float:
    """全年度の composite_score 平均を返す。"""
    scores = []
    for df in all_data.values():
        m = evaluate_single(df, params)
        scores.append(composite_score(m))
    return np.mean(scores)


# ------------------------------------------------------------------
# データ読み込み
# ------------------------------------------------------------------

def _parse_prize(x) -> float:
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    return float(str(x).replace(",", "").replace("万", "").strip())


def load_year(year: int) -> pd.DataFrame | None:
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        return None
    horses = pd.read_csv(csv_path)
    horses["prize_num"] = horses["total_prize"].apply(_parse_prize)
    features = build_feature_matrix(horses, birth_year=year)
    features["prize_num"] = horses["prize_num"].values
    return features


# ------------------------------------------------------------------
# グリッドサーチ
# ------------------------------------------------------------------

def grid_search(years):
    print("=== データ読み込み ===")
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

    # 現行パラメータ（model.py の値に合わせる）
    current_params = {
        "w_sire_ei": 0.225, "w_dam_prize": 0.075, "w_bms_ei": 0.25,
        "w_first_crop": 0.8,
        "b_early": 8, "b_parents_young": 8, "b_dam_bms_gap": 5,
        "b_sale_price": 5, "b_foal_penalty": 5, "b_foal_bonus": 3,
        "w_trainer": 0.05, "w_owner": 0.08, "w_breeder": 0.15,
        "dam_breed_base": 7, "dam_breed_cap": 4, "dam_breed_penalty": 2,
    }

    current_cv = cv_score(all_data, current_params)
    print(f"\n  現行パラメータのCVスコア: {current_cv:.2f}")
    for y, df in sorted(all_data.items()):
        m = evaluate_single(df, current_params)
        print(f"    {y}年: Spearman={m['spearman']:.4f} TOP10={m['top10']} TOP30={m['top30']} TOP100={m['top100']} 賞金倍率={m['prize_ratio']:.2f}x")

    best_score = current_cv
    best_params = dict(current_params)

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

    # ------------------------------------------------------------------
    # 最終結果
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  最適パラメータ")
    print("=" * 70)
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")
    print(f"\n  CVスコア: {best_score:.2f} (現行: {current_cv:.2f}, 差: {best_score - current_cv:+.2f})")

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
    print(f"{'年':>6} | {'指標':>10} | {'現行':>8} | {'最適化':>8} | {'差':>8}")
    print("-" * 60)
    for y in sorted(all_data.keys()):
        m_old = evaluate_single(all_data[y], current_params)
        m_new = evaluate_single(all_data[y], best_params)
        print(f"  {y} | {'Spearman':>10} | {m_old['spearman']:8.4f} | {m_new['spearman']:8.4f} | {m_new['spearman']-m_old['spearman']:+8.4f}")
        print(f"       | {'TOP10':>10} | {m_old['top10']:8d} | {m_new['top10']:8d} | {m_new['top10']-m_old['top10']:+8d}")
        print(f"       | {'TOP30':>10} | {m_old['top30']:8d} | {m_new['top30']:8d} | {m_new['top30']-m_old['top30']:+8d}")
        print(f"       | {'TOP100':>10} | {m_old['top100']:8d} | {m_new['top100']:8d} | {m_new['top100']-m_old['top100']:+8d}")
        print(f"       | {'賞金倍率':>10} | {m_old['prize_ratio']:8.2f}x | {m_new['prize_ratio']:8.2f}x | {m_new['prize_ratio']-m_old['prize_ratio']:+8.2f}")
        print("-" * 60)

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="グリッドサーチ（重み最適化）")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2019],
                        help="使用する年度リスト")
    args = parser.parse_args()

    best = grid_search(args.years)
