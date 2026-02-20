"""
追加特徴量データを一括取得するスクリプト。

馬一覧CSV取得後に実行し、以下を補完する:
  - 生年月日 → birth_dates_{year}.json
  - セリ価格・産駒番号 → extra_features_{year}.json

※ 親の生年（父・母・母父）は馬一覧取得時にCSVへ保存済みのため不要。

使い方:
  python scripts/fetch_all_features.py 2024
  python scripts/fetch_all_features.py 2024 --max-horses 100
  python scripts/fetch_all_features.py 2024 --top 500   # 上位500頭のみ
"""

import argparse
import sys
import os
import json
import threading

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper import (
    fetch_horse_profile,
    fetch_dam_foal_list,
    concurrent_fetch,
)


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _prescore(horses: pd.DataFrame) -> pd.Series:
    """CSVデータだけで計算できる暫定スコアを算出する。

    プロフィール取得前に上位候補を絞り込むために使用。
    父EI + 母馬賞金 + 母父EI + 調教師/馬主/牧場スコア + 親年齢ボーナス。
    """
    from src.features import (
        get_sire_ei, get_bms_ei, get_dam_prize,
        calc_trainer_score, calc_owner_score, calc_breeder_score,
        WEIGHT_SIRE_EI, WEIGHT_DAM_PRIZE, WEIGHT_BMS_EI, DAM_PRIZES,
    )

    scores = pd.Series(0.0, index=horses.index)

    # 父EI
    sire_ei = horses["sire"].apply(lambda x: get_sire_ei(x) if pd.notna(x) else 0.0)
    max_ei = sire_ei.max()
    if max_ei > 0:
        scores += (sire_ei / max_ei) * 100 * WEIGHT_SIRE_EI

    # 母父EI
    bms_ei = horses["sire_of_dam"].apply(lambda x: get_bms_ei(x) if pd.notna(x) else 0.0)
    max_bms = bms_ei.max()
    if max_bms > 0:
        scores += (bms_ei / max_bms) * 100 * WEIGHT_BMS_EI

    # 母馬賞金（不明なら0）
    raw_dp = horses["dam"].apply(lambda x: get_dam_prize(x) if pd.notna(x) else 0.0)
    dp_log = np.log1p(raw_dp)
    max_dp = dp_log.max()
    if max_dp > 0:
        scores += (dp_log / max_dp) * 100 * WEIGHT_DAM_PRIZE

    # 調教師・馬主・牧場
    scores += horses["trainer"].apply(lambda x: calc_trainer_score(x) - 50).fillna(0) * 0.2
    scores += horses["owner"].apply(lambda x: calc_owner_score(x) - 50).fillna(0) * 0.2
    scores += horses["breeder"].apply(lambda x: calc_breeder_score(x) - 50).fillna(0) * 0.3

    # 親年齢ボーナス（CSVに含まれている場合）
    for by_col in ["sire_birth_year", "dam_birth_year"]:
        if by_col in horses.columns:
            ages = horses.get("horse_id").apply(
                lambda x: int(str(x)[:4]) if str(x)[:4].isdigit() else None
            ) - pd.to_numeric(horses[by_col], errors="coerce")
            young = (ages <= 13) & ages.notna()
            scores += young.astype(float) * 2.5

    return scores


def fetch_all_features(birth_year: int, max_horses: int = None, top: int = None):
    csv_path = f"data/horses_{birth_year}.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} が見つかりません。先にデータを収集してください。")
        return

    horses = pd.read_csv(csv_path)
    if max_horses:
        horses = horses.head(max_horses)
    total = len(horses)

    # --- プレスコアで上位候補に絞り込み ---
    if top and top < total:
        print(f"=== プレスコア: {total}頭 → 上位{top}頭に絞り込み ===")
        horses["_prescore"] = _prescore(horses)
        horses = horses.nlargest(top, "_prescore").drop(columns=["_prescore"])
        print(f"  絞り込み完了: {len(horses)}頭")

    total = len(horses)
    print(f"\n=== {birth_year}年世代: {total}頭 ===\n")

    # キャッシュファイル
    bd_path = f"data/birth_dates_{birth_year}.json"
    ef_path = f"data/extra_features_{birth_year}.json"

    bd_cache = _load_cache(bd_path)    # {horse_id: "YYYY年MM月DD日"}
    ef_cache = _load_cache(ef_path)    # {horse_id: {sale_price: N, foal_number: N}}

    print(f"  既存キャッシュ: birth_dates={len(bd_cache)}, extra_features={len(ef_cache)}")

    # --- Phase 1: プロフィールページ（生年月日 + セリ価格） ---
    print(f"\n=== Phase 1: プロフィールデータ取得 ===")

    to_fetch_phase1 = []
    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        has_bd = hid in bd_cache
        has_ef = hid in ef_cache and "sale_price" in ef_cache.get(hid, {})
        if not (has_bd and has_ef):
            to_fetch_phase1.append(hid)

    print(f"  未取得: {len(to_fetch_phase1)}頭 (全{total}頭中)")

    if to_fetch_phase1:
        cache_lock = threading.Lock()

        def on_profile_result(hid, profile, _idx):
            with cache_lock:
                if profile is not None:
                    bd_cache[hid] = profile["birth_date"] or ""
                    if hid not in ef_cache:
                        ef_cache[hid] = {}
                    ef_cache[hid]["sale_price"] = profile["sale_price"]
                else:
                    bd_cache[hid] = ""
                    if hid not in ef_cache:
                        ef_cache[hid] = {}
                    ef_cache[hid].setdefault("sale_price", None)

        def save_phase1():
            with cache_lock:
                _save_cache(bd_path, dict(bd_cache))
                _save_cache(ef_path, dict(ef_cache))

        concurrent_fetch(
            items=to_fetch_phase1,
            fetch_fn=fetch_horse_profile,
            label="プロフィール",
            on_result=on_profile_result,
            save_interval=50,
            save_fn=save_phase1,
        )
    else:
        print("  全てキャッシュ済み")

    print(f"  Phase 1完了")

    # --- Phase 2: 産駒番号（母馬ページから何番仔かを取得） ---
    print(f"\n=== Phase 2: 産駒番号（何番仔か）取得 ===")

    # 母馬産駒リストのファイルキャッシュ（年度横断で再利用可能）
    dam_foals_path = "data/dam_foals.json"
    dam_foals_cache = _load_cache(dam_foals_path)  # {dam_id: [horse_id, ...]}

    # CSVのdam_idカラムを使用（馬一覧取得時に保存済み）
    dams_to_fetch = set()
    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        entry = ef_cache.get(hid, {})
        dam_id = str(row.get("dam_id", "")) if pd.notna(row.get("dam_id")) else ""
        if dam_id and "foal_number" not in entry:
            if dam_id not in dam_foals_cache:
                dams_to_fetch.add(dam_id)
            if hid not in ef_cache:
                ef_cache[hid] = {}
            ef_cache[hid]["dam_id"] = dam_id

    print(f"  未取得の母馬: {len(dams_to_fetch)}頭 (キャッシュ済: {len(dam_foals_cache)})")

    if dams_to_fetch:
        dam_lock = threading.Lock()

        def on_dam_result(dam_id, foal_list, _idx):
            with dam_lock:
                dam_foals_cache[dam_id] = foal_list if foal_list is not None else []

        def save_dam_foals():
            with dam_lock:
                _save_cache(dam_foals_path, dict(dam_foals_cache))

        concurrent_fetch(
            items=list(dams_to_fetch),
            fetch_fn=fetch_dam_foal_list,
            label="母馬",
            on_result=on_dam_result,
            save_interval=50,
            save_fn=save_dam_foals,
        )

    for _, row in horses.iterrows():
        hid = str(row["horse_id"])
        entry = ef_cache.get(hid, {})
        dam_id = entry.get("dam_id") or (str(row.get("dam_id", "")) if pd.notna(row.get("dam_id")) else "")
        if dam_id and "foal_number" not in entry:
            foal_list = dam_foals_cache.get(dam_id, [])
            entry["foal_number"] = (foal_list.index(hid) + 1) if hid in foal_list else None
            ef_cache[hid] = entry

    _save_cache(ef_path, ef_cache)
    print(f"  Phase 2完了")

    # --- サマリー ---
    print(f"\n=== 完了 ===")
    bd_ok = sum(1 for v in bd_cache.values() if v)
    ef_price = sum(1 for v in ef_cache.values() if v.get("sale_price"))
    ef_foal = sum(1 for v in ef_cache.values() if v.get("foal_number"))
    print(f"  生年月日:  {bd_ok}/{total}")
    print(f"  セリ価格:  {ef_price}/{total}")
    print(f"  産駒番号:  {ef_foal}/{total}")
    print(f"  → {bd_path}, {ef_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="追加特徴量データを一括取得")
    parser.add_argument("year", type=int, help="対象の生年（例: 2024）")
    parser.add_argument("--max-horses", type=int, default=None, help="最大取得頭数")
    parser.add_argument("--top", type=int, default=None,
                        help="プレスコア上位N頭のみプロフィール取得（例: 500）")
    args = parser.parse_args()
    fetch_all_features(args.year, max_horses=args.max_horses, top=args.top)
