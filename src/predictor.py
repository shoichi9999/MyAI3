"""
POG 予測 - メイン実行モジュール。

ダービー（牡馬）またはオークス（牝馬）のTOP5入り候補を予測する。

使い方:
  python run.py --year 2024                  # ダービー予測（デフォルト）
  python run.py --year 2024 --race oaks      # オークス予測
  python run.py --mode collect --year 2024
  python run.py --mode predict --year 2024 --race oaks
"""

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from src.scraper import fetch_horse_list_by_year
from src.features import build_feature_matrix
from src.model import heuristic_score


def collect_data(birth_year: int, max_horses: int = None) -> pd.DataFrame:
    """データを収集してCSVに保存する。"""
    print(f"\n{'='*60}")
    print(f"  データ収集: {birth_year}年生まれの2歳馬")
    print(f"{'='*60}")

    max_pages = None if not max_horses else (max_horses // 100) + 1
    horse_list = fetch_horse_list_by_year(birth_year, max_pages=max_pages)

    if horse_list.empty:
        print("[WARN] 馬一覧の取得に失敗しました。")
        return horse_list

    if max_horses:
        horse_list = horse_list.head(max_horses)

    print(f"  取得完了: {len(horse_list)}頭")
    horse_list.to_csv(f"data/horses_{birth_year}.csv", index=False, encoding="utf-8-sig")
    return horse_list


def predict_top(
    target_year: int,
    top_n: int = 10,
    race_type: str = "derby",
) -> pd.DataFrame:
    """
    ヒューリスティックスコアでTOP N予測を行う。

    Parameters
    ----------
    target_year : int
        対象の生年
    top_n : int
        上位何頭を出力するか
    race_type : str
        "derby" でダービー（牡馬）、"oaks" でオークス（牝馬）

    Returns
    -------
    pd.DataFrame
        予測結果（上位N頭）
    """
    race_label = "ダービー" if race_type == "derby" else "オークス"
    sex_label = "牡馬" if race_type == "derby" else "牝馬"
    sex_filter = "牡" if race_type == "derby" else "牝"

    print(f"\n{'='*60}")
    print(f"  POG{sex_label}{race_label}予測: {target_year}年生まれ TOP{top_n}")
    print(f"{'='*60}")

    horses_path = f"data/horses_{target_year}.csv"

    if not os.path.exists(horses_path):
        print(f"[ERROR] {horses_path} が見つかりません。先にデータを収集してください。")
        return pd.DataFrame()

    horses_df = pd.read_csv(horses_path)
    features = build_feature_matrix(horses_df, birth_year=target_year, sex_filter=sex_filter)
    features["score"] = heuristic_score(features, race_type=race_type)

    # スコアでソートしてTOP N
    features = features.sort_values("score", ascending=False)
    top = features.head(top_n).copy()

    # 表示用に整形
    display_cols = [
        "horse_name", "score",
        "sire_ei", "bms_ei", "dam_prize",
        "birth_month", "sire_age", "dam_age",
    ]
    available_cols = [c for c in display_cols if c in top.columns]
    display_df = top[available_cols].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "順位"

    col_rename = {
        "horse_name": "馬名",
        "score": "スコア",
        "sire_ei": "父EI",
        "bms_ei": "母父EI",
        "dam_prize": "母馬賞金(万)",
        "birth_month": "生月",
        "sire_age": "父年齢",
        "dam_age": "母年齢",
    }
    display_df = display_df.rename(columns=col_rename)

    print("\n" + tabulate(display_df, headers="keys", tablefmt="grid", floatfmt=".1f"))

    # 結果をCSVに保存
    race_suffix = "_oaks" if race_type == "oaks" else ""
    output_path = f"data/pog_top{top_n}{race_suffix}_{target_year}.csv"
    top.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果を保存しました: {output_path}")

    return top


def _fetch_extra_features(target_year: int, max_horses: int = None, top: int = None):
    """追加特徴量（生年月日・セリ価格・産駒番号）を一括取得する。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fetch_all_features",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "fetch_all_features.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.fetch_all_features(target_year, max_horses=max_horses, top=top)


def _fetch_dam_prizes(target_year: int, prescore_top: int = 0):
    """母馬賞金データを取得する（未取得分のみ）。

    prescore_top > 0 の場合、簡易スコア（父EI+母父EI+調教師等）で
    上位候補の母馬のみ取得する（全母馬の取得を回避して大幅高速化）。
    """
    import importlib.util
    import json
    import threading
    from src.scraper import concurrent_fetch

    spec = importlib.util.spec_from_file_location(
        "fetch_dam_prizes",
        os.path.join(os.path.dirname(__file__), "..", "fetch_dam_prizes.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    csv_path = f"data/horses_{target_year}.csv"
    if not os.path.exists(csv_path):
        return

    horses = pd.read_csv(csv_path)
    total_horses = len(horses)

    # --- 簡易スコアで上位候補に絞り込み（dam_prizeなし） ---
    if prescore_top and prescore_top < total_horses:
        from src.features import (
            get_sire_ei, get_bms_ei,
            calc_trainer_score, calc_owner_score, calc_breeder_score,
            get_leading_year,
        )
        ly = get_leading_year(target_year)
        lite = pd.Series(0.0, index=horses.index)
        sei = horses["sire"].apply(lambda x: get_sire_ei(x, ly) if pd.notna(x) else 0.0)
        mx = sei.max()
        if mx > 0:
            lite += (sei / mx) * 100 * 0.40
        bei = horses["sire_of_dam"].apply(lambda x: get_bms_ei(x, ly) if pd.notna(x) else 0.0)
        mx2 = bei.max()
        if mx2 > 0:
            lite += (bei / mx2) * 100 * 0.20
        lite += horses["trainer"].apply(lambda x: calc_trainer_score(x) - 50).fillna(0) * 0.05
        lite += horses["owner"].apply(lambda x: calc_owner_score(x) - 50).fillna(0) * 0.05
        lite += horses["breeder"].apply(lambda x: calc_breeder_score(x) - 50).fillna(0) * 0.08

        n_candidates = min(prescore_top * 3, total_horses)
        horses = horses.loc[lite.nlargest(n_candidates).index]
        print(f"\n  母馬賞金: 簡易スコアで{n_candidates}頭に絞り込み（全{total_horses}頭中）")

    # dam_name → dam_id マッピング構築
    dam_info = {}  # {name: dam_id or None}
    for _, row in horses.iterrows():
        name = row.get("dam")
        did = row.get("dam_id")
        if pd.notna(name) and str(name).strip():
            n = str(name).strip()
            if n not in dam_info:
                dam_info[n] = str(did).strip() if pd.notna(did) and str(did).strip() else None

    cache = {}
    if os.path.exists(mod.CACHE_FILE):
        with open(mod.CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # 産駒リストキャッシュも同時に更新（Phase 2の重複リクエストを排除）
    dam_foals_path = "data/dam_foals.json"
    dam_foals_cache = {}
    if os.path.exists(dam_foals_path):
        with open(dam_foals_path, "r", encoding="utf-8") as f:
            dam_foals_cache = json.load(f)

    to_fetch = [d for d in sorted(dam_info.keys()) if d not in cache]
    if not to_fetch:
        print(f"\n--- 母馬賞金: 全{len(dam_info)}頭キャッシュ済み。スキップ ---")
        return

    id_count = sum(1 for d in to_fetch if dam_info.get(d))
    print(f"\n{'='*60}")
    print(f"  母馬賞金+産駒リスト取得: 新規{len(to_fetch)}頭（ID直接: {id_count}, 名前検索: {len(to_fetch) - id_count}）")
    print(f"{'='*60}")

    def fetch_combined(name):
        """dam_idがあれば1リクエストで賞金+産駒リストを同時取得。"""
        did = dam_info.get(name)
        if did:
            return mod.fetch_dam_prize_and_foals(did)
        # IDなし: 名前検索（産駒リストは取れない）
        prize = mod.fetch_horse_prize_by_name(name)
        return {"prize": prize, "foals": None}

    cache_lock = threading.Lock()

    def on_result(name, result, _idx):
        with cache_lock:
            if result is None:
                cache[name] = 0.0
                return
            cache[name] = result["prize"] if result["prize"] is not None else 0.0
            # 産駒リストもキャッシュに保存
            did = dam_info.get(name)
            if did and result.get("foals") is not None:
                dam_foals_cache[did] = result["foals"]

    def save_fn():
        with cache_lock:
            with open(mod.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(dict(cache), f, ensure_ascii=False, indent=2)
            with open(dam_foals_path, "w", encoding="utf-8") as f:
                json.dump(dict(dam_foals_cache), f, ensure_ascii=False, indent=2)

    concurrent_fetch(
        items=to_fetch,
        fetch_fn=fetch_combined,
        label="母馬賞金+産駒",
        on_result=on_result,
        save_interval=100,
        save_fn=save_fn,
    )
    print(f"  完了: 賞金{len(cache)}頭, 産駒リスト{len(dam_foals_cache)}頭")


def run_full_pipeline(
    target_year: int = 2024,
    max_horses: int = None,
    top_n: int = 10,
    prescore_top: int = 500,
    race_type: str = "derby",
):
    """
    全自動パイプライン（データ収集 → 特徴量取得 → 予測）。

    Parameters
    ----------
    target_year : int
        予測対象の世代（生年）
    max_horses : int
        取得する最大馬数（テスト用）
    top_n : int
        上位何頭を出力するか
    prescore_top : int
        プレスコア上位N頭のみプロフィール取得（0で全頭取得）
    race_type : str
        "derby" でダービー（牡馬）、"oaks" でオークス（牝馬）
    """
    race_label = "ダービー" if race_type == "derby" else "オークス"
    sex_label = "牡馬" if race_type == "derby" else "牝馬"

    print("=" * 60)
    print(f"  POG{sex_label}{race_label}予測 - 全自動パイプライン")
    print(f"  対象世代: {target_year}年生まれ（{sex_label}のみ）")
    print(f"  予測馬数: TOP{top_n}")
    if prescore_top:
        print(f"  プレスコア絞り込み: 上位{prescore_top}頭")
    print("=" * 60)

    # 1. 馬一覧データ収集
    if not os.path.exists(f"data/horses_{target_year}.csv"):
        collect_data(target_year, max_horses=max_horses)
    else:
        print(f"\n--- {target_year}年世代: 馬一覧データ既存。スキップ ---")

    # 2. 母馬賞金（先に取得してDAM_PRIZESを更新→プレスコアで利用）
    _fetch_dam_prizes(target_year, prescore_top=prescore_top)

    # DAM_PRIZESをリロード（モジュールレベル変数は初回import時の値のまま）
    import src.features as _features_mod
    _features_mod.DAM_PRIZES = _features_mod._load_json("data/dam_prizes.json")

    # 3. 追加特徴量（生年月日・セリ価格・産駒番号）
    _fetch_extra_features(target_year, max_horses=max_horses, top=prescore_top or None)

    # 4. 予測
    result = predict_top(target_year, top_n=top_n, race_type=race_type)

    return result


def main():
    parser = argparse.ArgumentParser(description="POG予測システム（ダービー/オークス）")
    parser.add_argument(
        "--mode",
        choices=["collect", "predict", "full"],
        default="full",
        help="実行モード (default: full)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="対象の生年 (default: 2024)",
    )
    parser.add_argument(
        "--max-horses",
        type=int,
        default=None,
        help="取得する最大馬数（テスト用）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="上位何頭を出力するか (default: 10)",
    )
    parser.add_argument(
        "--prescore-top",
        type=int,
        default=500,
        help="プレスコア上位N頭のみプロフィール取得 (default: 500, 0=全頭)",
    )
    parser.add_argument(
        "--race",
        choices=["derby", "oaks"],
        default="derby",
        help="対象レース: derby(ダービー・牡馬) / oaks(オークス・牝馬)",
    )

    args = parser.parse_args()

    if args.mode == "collect":
        collect_data(args.year, max_horses=args.max_horses)
    elif args.mode == "predict":
        predict_top(args.year, top_n=args.top_n, race_type=args.race)
    elif args.mode == "full":
        run_full_pipeline(
            target_year=args.year,
            max_horses=args.max_horses,
            top_n=args.top_n,
            prescore_top=args.prescore_top,
            race_type=args.race,
        )


if __name__ == "__main__":
    main()
