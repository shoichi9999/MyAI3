"""
特徴量エンジニアリングモジュール。

POG予測に重要な特徴量を生成する（デビュー前に入手可能な情報のみ）:
- 血統スコア（父馬の産駒EI、母父馬の産駒EI、母馬の獲得賞金）
- 生まれ月（早生まれほど有利）
- 親年齢（父・母の産駒時年齢）
- 調教師・馬主・牧場スコア（実績ベース）
- セリ価格・産駒番号
"""

import json
import os
import re

import numpy as np
import pandas as pd


# === リーディングデータ（産駒成績ベース） ===

def _load_json(path: str) -> dict:
    """JSONファイルを読み込む。存在しなければ空辞書を返す。"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 年度別リーディングキャッシュ
_LEADING_CACHE: dict[tuple[str, int], dict] = {}

# デフォルト（最新）のリーディングデータ — 予測時に使用
_DEFAULT_LEADING_YEAR = 2024
SIRE_LEADING = _load_json("data/sire_leading_2024.json")
BMS_LEADING = _load_json("data/bms_leading_2024.json")


def _load_leading(kind: str, year: int) -> dict:
    """年度別リーディングデータを読み込む（キャッシュ付き）。

    Parameters
    ----------
    kind : str
        "sire_leading" or "bms_leading"
    year : int
        リーディングの年度

    Returns
    -------
    dict
        {馬名: {rank, progeny_prize, ei, ...}}
    """
    key = (kind, year)
    if key not in _LEADING_CACHE:
        _LEADING_CACHE[key] = _load_json(f"data/{kind}_{year}.json")
    return _LEADING_CACHE[key]


def get_leading_year(birth_year: int) -> int:
    """POGドラフト時点で利用可能なリーディング年度を返す。

    生年Yの馬 → デビューはY+2年 → POGドラフトはY+2年春
    → 最新の通年データは Y+1 年。
    ただし該当ファイルがなければ最も近い年に降格。
    """
    target = birth_year + 1
    # 対象年のファイルが存在するかチェック
    for y in [target, target - 1, target + 1, _DEFAULT_LEADING_YEAR]:
        if os.path.exists(f"data/sire_leading_{y}.json"):
            return y
    return _DEFAULT_LEADING_YEAR
# クラシック結果（兄姉実績・種牡馬クラシック輩出数の計算用）
_CLASSIC_RESULTS = _load_json("data/classic_results.json")

# 母馬の獲得賞金
DAM_PRIZES = _load_json("data/dam_prizes.json")
# 種牡馬自身の現役獲得賞金（初年度種牡馬ボーナス用）
SIRE_PRIZES = _load_json("data/sire_prizes.json")
# 母馬産駒リスト（dam_id → [horse_id, ...]、生年順）
DAM_FOALS = _load_json("data/dam_foals.json")
# 生年月日キャッシュ（世代別）
BIRTH_DATES_CACHE = {}
# 親年齢キャッシュ（世代別）
PARENT_AGES_CACHE = {}
# 追加特徴量キャッシュ（セリ価格・産駒番号、世代別）
EXTRA_FEATURES_CACHE = {}


def _load_birth_dates(birth_year: int) -> dict:
    """生年月日キャッシュを読み込む。"""
    if birth_year not in BIRTH_DATES_CACHE:
        BIRTH_DATES_CACHE[birth_year] = _load_json(f"data/birth_dates_{birth_year}.json")
    return BIRTH_DATES_CACHE[birth_year]


def _load_parent_ages(birth_year: int) -> dict:
    """親年齢キャッシュを読み込む。"""
    if birth_year not in PARENT_AGES_CACHE:
        PARENT_AGES_CACHE[birth_year] = _load_json(f"data/parent_ages_{birth_year}.json")
    return PARENT_AGES_CACHE[birth_year]


def _load_extra_features(birth_year: int) -> dict:
    """追加特徴量キャッシュ（セリ価格・産駒番号）を読み込む。"""
    if birth_year not in EXTRA_FEATURES_CACHE:
        EXTRA_FEATURES_CACHE[birth_year] = _load_json(f"data/extra_features_{birth_year}.json")
    return EXTRA_FEATURES_CACHE[birth_year]


# エリートデータ・重み設定を外部JSONから読み込み（data/config/）
ELITE_TRAINERS = _load_json("data/config/elite_trainers.json")
ELITE_TRAINERS.pop("_comment", None)
ELITE_OWNERS = _load_json("data/config/elite_owners.json")
ELITE_OWNERS.pop("_comment", None)
ELITE_BREEDERS = _load_json("data/config/elite_breeders.json")
ELITE_BREEDERS.pop("_comment", None)

_WEIGHTS = _load_json("data/config/weights.json")
WEIGHT_SIRE_EI = _WEIGHTS.get("w_sire_ei", 0.065)
WEIGHT_DAM_PRIZE = _WEIGHTS.get("w_dam_prize", 0.036)
WEIGHT_BMS_EI = _WEIGHTS.get("w_bms_ei", 0.0235)


# ------------------------------------------------------------------
# リーク防止: 時点制約付きクラシック実績
# ------------------------------------------------------------------

def _get_classic_ids_up_to(max_birth_year: int) -> set:
    """max_birth_year 以前の世代のクラシック馬IDセットを返す。

    時点制約: 生年Yの馬 → ダービー/オークスは Y+3年春。
    POGドラフト(birth_year+2年春)時点で結果が確定しているのは
    birth_year-2 世代以前（= Y+2年より前のレース結果）。
    呼び出し側で適切な max_birth_year を渡すこと。
    """
    ids = set()
    for race in ("derby", "oaks"):
        for by_str, horse_ids in _CLASSIC_RESULTS.get(race, {}).items():
            if int(by_str) <= max_birth_year:
                ids.update(horse_ids)
    return ids


# 種牡馬クラシック輩出数キャッシュ（max_birth_year → {sire_name: count}）
_SIRE_CLASSIC_CACHE: dict[int, dict[str, int]] = {}


def _get_sire_classic_map(max_birth_year: int) -> dict[str, int]:
    """種牡馬ごとのクラシックTOP5輩出数を返す（リーク防止: max_birth_year以前のみ）。"""
    if max_birth_year in _SIRE_CLASSIC_CACHE:
        return _SIRE_CLASSIC_CACHE[max_birth_year]

    sire_counts: dict[str, int] = {}
    for race in ("derby", "oaks"):
        for by_str, horse_ids in _CLASSIC_RESULTS.get(race, {}).items():
            by = int(by_str)
            if by > max_birth_year:
                continue
            csv_path = f"data/horses_{by}.csv"
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            id_to_sire = dict(zip(
                df["horse_id"].astype(str), df["sire"].fillna("")
            ))
            for hid in horse_ids:
                sire = id_to_sire.get(str(hid), "")
                if sire:
                    sire_counts[sire] = sire_counts.get(sire, 0) + 1

    _SIRE_CLASSIC_CACHE[max_birth_year] = sire_counts
    return sire_counts


def get_sire_2yo_ei(sire_name: str, leading_year: int = None) -> float:
    """種牡馬の2歳EI（2歳産駒限定のアーニングインデックス）を返す。

    data/sire_2yo_leading_{year}.json が存在する場合のみ有効。
    """
    if not sire_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("sire_2yo_leading", leading_year).get(sire_name, {})
    else:
        data = _load_json("data/sire_2yo_leading_2024.json").get(sire_name, {})
    try:
        return float(data.get("ei", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_sire_ei(sire_name: str, leading_year: int = None) -> float:
    """種牡馬のEI（アーニングインデックス）を返す。

    Parameters
    ----------
    sire_name : str
        種牡馬名
    leading_year : int, optional
        参照するリーディング年度。Noneならデフォルト（最新）を使用。
    """
    if not sire_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("sire_leading", leading_year).get(sire_name, {})
    else:
        data = SIRE_LEADING.get(sire_name, {})
    try:
        return float(data.get("ei", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_bms_ei(bms_name: str, leading_year: int = None) -> float:
    """母父馬のEI（アーニングインデックス）を返す。

    Parameters
    ----------
    bms_name : str
        母父馬名
    leading_year : int, optional
        参照するリーディング年度。Noneならデフォルト（最新）を使用。
    """
    if not bms_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("bms_leading", leading_year).get(bms_name, {})
    else:
        data = BMS_LEADING.get(bms_name, {})
    try:
        return float(data.get("ei", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_dam_prize(dam_name: str) -> float:
    """母馬の獲得賞金（万円）を返す。"""
    if not dam_name:
        return 0.0
    return DAM_PRIZES.get(dam_name, 0.0)


def get_sire_prize(sire_name: str) -> float:
    """種牡馬自身の現役獲得賞金（万円）を返す。"""
    if not sire_name:
        return 0.0
    return SIRE_PRIZES.get(sire_name, 0.0)


def calc_trainer_score(trainer_name: str) -> float:
    """調教師スコアを返す。"""
    if not trainer_name:
        return 50.0
    for key, score in ELITE_TRAINERS.items():
        if key in str(trainer_name):
            return score
    return 50.0


def calc_owner_score(owner_name: str) -> float:
    """馬主スコアを返す。"""
    if not owner_name:
        return 50.0
    for key, score in ELITE_OWNERS.items():
        if key in str(owner_name):
            return score
    return 50.0


def calc_breeder_score(breeder_name: str) -> float:
    """生産牧場スコアを返す。"""
    if not breeder_name:
        return 50.0
    for key, score in ELITE_BREEDERS.items():
        if key in str(breeder_name):
            return score
    return 50.0


def get_birth_month(horse_id: str, birth_year: int) -> int:
    """馬の生まれ月を返す（1-12）。取得できない場合は3（中央値）。"""
    bd_cache = _load_birth_dates(birth_year)
    bd = bd_cache.get(str(horse_id), "")
    if bd:
        m = re.search(r"(\d+)月", bd)
        if m:
            return int(m.group(1))
    return 3  # デフォルト: 3月


def _birth_year_from_id(horse_id: str) -> int | None:
    """horse_idの先頭4桁から生年を返す（日本産馬のみ）。"""
    if horse_id and str(horse_id)[:4].isdigit():
        return int(str(horse_id)[:4])
    return None


# 親の (CSVカラム名, JSONキャッシュのbirth_yearキー, JSONキャッシュのidキー)
_PARENT_MAP = {
    "sire": ("sire_birth_year", "sire_birth_year", "sire_id"),
    "dam":  ("dam_birth_year",  "dam_birth_year",  "dam_id"),
    "bms":  ("bms_birth_year",  "dam_sire_birth_year", "bms_id"),
}


def get_parent_age(horse_row, birth_year: int, parent: str = "sire") -> float | None:
    """親の産駒時年齢を返す。取得できない場合はNone。

    優先順:
      1. CSVカラム (sire_birth_year 等) から直接計算
      2. JSONキャッシュ (parent_ages_{year}.json) からフォールバック
    """
    csv_col, cache_by_key, cache_id_key = _PARENT_MAP.get(
        parent, (f"{parent}_birth_year", f"{parent}_birth_year", None)
    )

    # 1. CSVカラムから（DataFrameの行に含まれている場合）
    parent_by = horse_row.get(csv_col)
    if pd.notna(parent_by) and parent_by:
        try:
            return birth_year - int(parent_by)
        except (ValueError, TypeError):
            pass

    # 2. JSONキャッシュからフォールバック
    hid = str(horse_row.get("horse_id", ""))
    pa_cache = _load_parent_ages(birth_year)
    data = pa_cache.get(hid, {})

    by = data.get(cache_by_key)
    if by:
        return birth_year - by

    if cache_id_key:
        parent_id = data.get(cache_id_key)
        if parent_id:
            parent_by = _birth_year_from_id(parent_id)
            if parent_by:
                return birth_year - parent_by

    return None


def build_feature_matrix(horses_df: pd.DataFrame, birth_year: int = None) -> pd.DataFrame:
    """
    全馬の特徴量マトリクスを構築する。

    血統スコア（父EI、母父EI、母馬獲得賞金）、生まれ月、親年齢を
    使って予測に使える特徴量を生成する。

    EIデータは birth_year に対応するリーディング年度
    （= birth_year + 1、POGドラフト時点で利用可能な最新データ）を参照する。
    """
    feature_rows = []

    # birth_yearの推定（horse_idの先頭4桁 or DataFrameから）
    if birth_year is None:
        sample_id = str(horses_df.iloc[0].get("horse_id", ""))
        if sample_id[:4].isdigit():
            birth_year = int(sample_id[:4])
        else:
            birth_year = 2024

    # リーディング年度の決定（データリーク防止）
    leading_year = get_leading_year(birth_year)

    for _, horse in horses_df.iterrows():
        hid = horse.get("horse_id", "")
        row = {"horse_id": hid}

        # 基本情報
        row["horse_name"] = horse.get("horse_name", "")
        row["sex"] = 1 if horse.get("sex") == "牡" else (0 if horse.get("sex") == "牝" else 0.5)

        # 産駒成績ベースの血統スコア（年度別リーディングを参照）
        row["sire_ei"] = get_sire_ei(horse.get("sire", ""), leading_year)
        row["bms_ei"] = get_bms_ei(horse.get("sire_of_dam", ""), leading_year)
        row["sire_2yo_ei"] = get_sire_2yo_ei(horse.get("sire", ""), leading_year)

        # 母馬の獲得賞金（不明なら0 — 不明は不利な情報として扱う）
        row["dam_prize"] = get_dam_prize(horse.get("dam", ""))

        # 種牡馬自身の現役獲得賞金（初年度種牡馬ボーナス用）
        row["sire_prize"] = get_sire_prize(horse.get("sire", ""))

        # 調教師・馬主・生産者スコア
        row["trainer_score"] = calc_trainer_score(horse.get("trainer", ""))
        row["owner_score"] = calc_owner_score(horse.get("owner", ""))
        row["breeder_score"] = calc_breeder_score(horse.get("breeder", ""))

        # 生まれ月（1-12、小さいほど有利）
        row["birth_month"] = get_birth_month(hid, birth_year)

        # 親年齢（父・母・母父の産駒時年齢）— 欠損はNaN（デフォルト値補完しない）
        sire_age = get_parent_age(horse, birth_year, "sire")
        dam_age = get_parent_age(horse, birth_year, "dam")
        dam_sire_age = get_parent_age(horse, birth_year, "bms")
        row["sire_age"] = sire_age if sire_age is not None else np.nan
        row["dam_age"] = dam_age if dam_age is not None else np.nan
        row["dam_sire_age"] = dam_sire_age if dam_sire_age is not None else np.nan

        # --- ドメイン知識ベースのバイナリ特徴量 ---
        # 年齢がNaNの場合はバイナリもNaN（デフォルト値で偽装しない）
        row["early_born"] = 1 if row["birth_month"] <= 4 else 0
        row["sire_young"] = (1 if sire_age <= 13 else 0) if sire_age is not None else np.nan
        row["dam_young"] = (1 if dam_age <= 13 else 0) if dam_age is not None else np.nan
        row["both_parents_young"] = (
            (1 if sire_age <= 13 and dam_age <= 13 else 0)
            if sire_age is not None and dam_age is not None else np.nan
        )
        row["sire_first_crop"] = (1 if sire_age <= 7 else 0) if sire_age is not None else np.nan
        row["dam_bms_gap_small"] = (
            (1 if (dam_sire_age - dam_age) <= 15 else 0)
            if dam_sire_age is not None and dam_age is not None else np.nan
        )

        # --- 追加特徴量（セリ価格・産駒番号） ---
        extra = _load_extra_features(birth_year).get(str(hid), {})
        sale_price = extra.get("sale_price")
        row["sale_price_log"] = np.log1p(sale_price) if sale_price else 0.0
        foal_number = extra.get("foal_number")
        # extra_featuresにない場合、dam_foalsから直接計算
        if not foal_number:
            dam_id_fn = str(horse.get("dam_id", "")) if pd.notna(horse.get("dam_id")) else ""
            if dam_id_fn:
                fl = DAM_FOALS.get(dam_id_fn, [])
                if fl and str(hid) in fl:
                    foal_number = len(fl) - fl.index(str(hid))
        row["foal_number"] = foal_number if foal_number else 3.0  # デフォルト: 3番仔

        # --- 母馬の繁殖入り年齢（初仔生年 - 母馬生年 = 引退時期の近似） ---
        dam_breeding_age = np.nan
        dam_by = horse.get("dam_birth_year")
        dam_id = str(horse.get("dam_id", "")) if pd.notna(horse.get("dam_id")) else ""
        foal_list = DAM_FOALS.get(dam_id, []) if dam_id else []
        if dam_id and pd.notna(dam_by):
            if foal_list:
                # 産駒リストは降順（新しい順）→ 末尾が初仔
                first_foal_by = _birth_year_from_id(foal_list[-1])
                if first_foal_by is not None:
                    dam_breeding_age = first_foal_by - int(dam_by)
        row["dam_breeding_age"] = dam_breeding_age

        # --- 母馬の総産駒数（リーク防止: birth_year以前に生まれた産駒のみカウント） ---
        if foal_list:
            known_foals = [f for f in foal_list
                           if (_birth_year_from_id(f) or 9999) <= birth_year]
            row["total_dam_foals"] = len(known_foals)
        else:
            row["total_dam_foals"] = 0

        # --- 兄姉のクラシック実績（リーク防止: birth_year-2以前の結果のみ） ---
        # POGドラフト(Y+2春)時点で確定 = (Y-2)世代以前のダービー/オークス
        classic_cutoff = birth_year - 2
        classic_ids = _get_classic_ids_up_to(classic_cutoff)
        older_siblings = [f for f in foal_list
                          if (_birth_year_from_id(f) or 9999) < birth_year]
        row["sibling_classic"] = (
            1 if any(str(s) in classic_ids for s in older_siblings) else 0
        )

        # --- 種牡馬のクラシックTOP5輩出数（リーク防止: birth_year-2以前のみ） ---
        sire_classic_map = _get_sire_classic_map(classic_cutoff)
        sire_name = horse.get("sire", "")
        row["sire_classic_count"] = sire_classic_map.get(sire_name, 0)

        # --- 特徴量交互作用（非線形シグナル） ---
        # 父EI × 母賞金: 良血父 × 良血母のシナジー
        sire_ei_val = row["sire_ei"]
        dam_prize_val = row["dam_prize"]
        row["sire_dam_interaction"] = sire_ei_val * np.log1p(dam_prize_val)

        # 調教師 × 生産者: エリート牧場→エリート調教師パイプライン
        row["trainer_breeder_combo"] = row["trainer_score"] * row["breeder_score"]

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)
