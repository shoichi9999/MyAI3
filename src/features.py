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
    ただし該当ファイルがなければ過去の年に降格（未来データは使わない）。
    """
    target = birth_year + 1
    # リーク防止: target以前のデータのみ使用（target+1 や DEFAULT は未来データの恐れ）
    for y in [target, target - 1, target - 2]:
        if os.path.exists(f"data/sire_leading_{y}.json"):
            return y
    return target  # ファイルがなくてもtargetを返す（データなし扱い）
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
# 追加特徴量キャッシュ（セリ価格・産駒番号、世代別）
EXTRA_FEATURES_CACHE = {}


def _load_birth_dates(birth_year: int) -> dict:
    """生年月日キャッシュを読み込む。"""
    if birth_year not in BIRTH_DATES_CACHE:
        BIRTH_DATES_CACHE[birth_year] = _load_json(f"data/birth_dates_{birth_year}.json")
    return BIRTH_DATES_CACHE[birth_year]


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

    ダービー・オークス両方を含めることで、兄姉・母の産駒品質の
    評価精度を向上させる。
    """
    ids = set()
    for race_key in ["derby", "oaks"]:
        for by_str, horse_ids in _CLASSIC_RESULTS.get(race_key, {}).items():
            if int(by_str) <= max_birth_year:
                ids.update(horse_ids)
    return ids


# 汎用クラシック輩出数キャッシュ: (max_birth_year, role, race_keys) → {name: count}
_CLASSIC_MAP_CACHE: dict[tuple, dict[str, int]] = {}

# CSVファイル読み込みキャッシュ（同じCSVを何度も読まない）
_CSV_CACHE: dict[str, pd.DataFrame] = {}


def _read_csv_cached(path: str) -> pd.DataFrame | None:
    """CSVファイルをキャッシュ付きで読み込む。"""
    if path not in _CSV_CACHE:
        if not os.path.exists(path):
            _CSV_CACHE[path] = None
        else:
            _CSV_CACHE[path] = pd.read_csv(path)
    return _CSV_CACHE[path]


def _build_classic_map(max_birth_year: int, column: str,
                       race_keys: tuple[str, ...]) -> dict[str, int]:
    """汎用クラシックTOP5輩出数マップを構築する（リーク防止: max_birth_year以前のみ）。

    Parameters
    ----------
    max_birth_year : int
        この年以前の世代のクラシック結果のみ使用
    column : str
        CSVの参照カラム名（"sire" or "sire_of_dam"）
    race_keys : tuple[str, ...]
        参照するレース種別のタプル（("derby", "oaks") or ("oaks",)）
    """
    cache_key = (max_birth_year, column, race_keys)
    if cache_key in _CLASSIC_MAP_CACHE:
        return _CLASSIC_MAP_CACHE[cache_key]

    counts: dict[str, int] = {}
    for race_key in race_keys:
        for by_str, horse_ids in _CLASSIC_RESULTS.get(race_key, {}).items():
            if int(by_str) > max_birth_year:
                continue
            df = _read_csv_cached(f"data/horses_{by_str}.csv")
            if df is None:
                continue
            id_to_name = dict(zip(
                df["horse_id"].astype(str), df[column].fillna("")
            ))
            for hid in horse_ids:
                name = id_to_name.get(str(hid), "")
                if name:
                    counts[name] = counts.get(name, 0) + 1

    _CLASSIC_MAP_CACHE[cache_key] = counts
    return counts


def _get_sire_classic_map(max_birth_year: int) -> dict[str, int]:
    """種牡馬ごとのクラシックTOP5輩出数を返す（ダービー+オークス）。"""
    return _build_classic_map(max_birth_year, "sire", ("derby", "oaks"))


def _get_sire_oaks_map(max_birth_year: int) -> dict[str, int]:
    """種牡馬ごとのオークスTOP5輩出数を返す（オークスのみ）。"""
    return _build_classic_map(max_birth_year, "sire", ("oaks",))


def _get_bms_classic_map(max_birth_year: int) -> dict[str, int]:
    """母父ごとのクラシックTOP5輩出数を返す（ダービー+オークス）。"""
    return _build_classic_map(max_birth_year, "sire_of_dam", ("derby", "oaks"))


def _get_sire_derby_map(max_birth_year: int) -> dict[str, int]:
    """種牡馬ごとのダービーTOP5輩出数を返す（ダービーのみ）。"""
    return _build_classic_map(max_birth_year, "sire", ("derby",))


def _get_bms_oaks_map(max_birth_year: int) -> dict[str, int]:
    """母父ごとのオークスTOP5輩出数を返す（オークスのみ）。"""
    return _build_classic_map(max_birth_year, "sire_of_dam", ("oaks",))


def _get_bms_derby_map(max_birth_year: int) -> dict[str, int]:
    """母父ごとのダービーTOP5輩出数を返す（ダービーのみ）。"""
    return _build_classic_map(max_birth_year, "sire_of_dam", ("derby",))


def _get_dam_classic_ids(max_birth_year: int) -> set:
    """母馬がオークスTOP5だったかの判定用ID set。"""
    ids = set()
    for by_str, horse_ids in _CLASSIC_RESULTS.get("oaks", {}).items():
        if int(by_str) <= max_birth_year:
            ids.update(str(hid) for hid in horse_ids)
    return ids


def get_sire_runners(sire_name: str, leading_year: int = None) -> int:
    """種牡馬の産駒出走頭数を返す。"""
    if not sire_name:
        return 0
    if leading_year is not None:
        data = _load_leading("sire_leading", leading_year).get(sire_name, {})
    else:
        data = SIRE_LEADING.get(sire_name, {})
    try:
        return int(data.get("runners", 0) or 0)
    except (ValueError, TypeError):
        return 0


def get_sire_progeny_prize(sire_name: str, leading_year: int = None) -> float:
    """種牡馬の産駒総賞金を返す。"""
    if not sire_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("sire_leading", leading_year).get(sire_name, {})
    else:
        data = SIRE_LEADING.get(sire_name, {})
    try:
        return float(data.get("progeny_prize", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_bms_runners(bms_name: str, leading_year: int = None) -> int:
    """母父馬の産駒出走頭数を返す。"""
    if not bms_name:
        return 0
    if leading_year is not None:
        data = _load_leading("bms_leading", leading_year).get(bms_name, {})
    else:
        data = BMS_LEADING.get(bms_name, {})
    try:
        return int(data.get("runners", 0) or 0)
    except (ValueError, TypeError):
        return 0


def get_sire_win_rate(sire_name: str, leading_year: int = None) -> float:
    """種牡馬の勝率（産駒の勝ち上がり率）を返す。"""
    if not sire_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("sire_leading", leading_year).get(sire_name, {})
    else:
        data = SIRE_LEADING.get(sire_name, {})
    try:
        return float(data.get("win_rate", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_bms_win_rate(bms_name: str, leading_year: int = None) -> float:
    """母父馬の勝率（産駒の勝ち上がり率）を返す。"""
    if not bms_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("bms_leading", leading_year).get(bms_name, {})
    else:
        data = BMS_LEADING.get(bms_name, {})
    try:
        return float(data.get("win_rate", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_bms_rank(bms_name: str, leading_year: int = None) -> int:
    """母父馬のリーディング順位を返す（ランク外は0）。"""
    if not bms_name:
        return 0
    if leading_year is not None:
        data = _load_leading("bms_leading", leading_year).get(bms_name, {})
    else:
        data = BMS_LEADING.get(bms_name, {})
    try:
        return int(data.get("rank", 0) or 0)
    except (ValueError, TypeError):
        return 0


def get_bms_progeny_prize(bms_name: str, leading_year: int = None) -> float:
    """母父馬の産駒総賞金を返す。"""
    if not bms_name:
        return 0.0
    if leading_year is not None:
        data = _load_leading("bms_leading", leading_year).get(bms_name, {})
    else:
        data = BMS_LEADING.get(bms_name, {})
    try:
        return float(data.get("progeny_prize", 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def get_sire_rank(sire_name: str, leading_year: int = None) -> int:
    """種牡馬のリーディング順位を返す（ランク外は0）。"""
    if not sire_name:
        return 0
    if leading_year is not None:
        data = _load_leading("sire_leading", leading_year).get(sire_name, {})
    else:
        data = SIRE_LEADING.get(sire_name, {})
    try:
        return int(data.get("rank", 0) or 0)
    except (ValueError, TypeError):
        return 0


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


# 親キー → CSVカラム名
_PARENT_MAP = {
    "sire": ("sire_birth_year",),
    "dam":  ("dam_birth_year",),
    "bms":  ("bms_birth_year",),
}




def _leading_lookup_series(ld_dict: dict, field: str) -> pd.Series:
    """リーディングデータから特定フィールドのルックアップ用Seriesを構築する。"""
    return pd.Series({
        k: float(v.get(field, 0) or 0) if isinstance(v, dict) else 0.0
        for k, v in ld_dict.items()
    })


def _leading_lookup_series_int(ld_dict: dict, field: str) -> pd.Series:
    """リーディングデータから特定フィールドのルックアップ用Series（int型）を構築する。"""
    return pd.Series({
        k: int(v.get(field, 0) or 0) if isinstance(v, dict) else 0
        for k, v in ld_dict.items()
    })


def build_feature_matrix(horses_df: pd.DataFrame, birth_year: int = None,
                         sex_filter: str = "牡") -> pd.DataFrame:
    """
    全馬の特徴量マトリクスを構築する（ベクトル化版）。

    血統スコア（父EI、母父EI、母馬獲得賞金）、生まれ月、親年齢を
    使って予測に使える特徴量を生成する。

    EIデータは birth_year に対応するリーディング年度
    （= birth_year + 1、POGドラフト時点で利用可能な最新データ）を参照する。

    Parameters
    ----------
    sex_filter : str
        "牡" でダービー用（牡馬のみ）、"牝" でオークス用（牝馬のみ）。
    """
    # birth_yearの推定（horse_idの先頭4桁 or DataFrameから）
    if birth_year is None:
        sample_id = str(horses_df.iloc[0].get("horse_id", ""))
        if sample_id[:4].isdigit():
            birth_year = int(sample_id[:4])
        else:
            birth_year = 2024

    # 性別フィルタリング（ダービー=牡馬、オークス=牝馬）
    src = horses_df[horses_df["sex"] == sex_filter].copy()

    # リーディング年度の決定（データリーク防止）
    leading_year = get_leading_year(birth_year)

    # --- データの事前ロード ---
    sire_ld = _load_leading("sire_leading", leading_year)
    bms_ld = _load_leading("bms_leading", leading_year)
    prev_sire_ld = _load_leading("sire_leading", leading_year - 1)
    bd_cache = _load_birth_dates(birth_year)
    ex_cache = _load_extra_features(birth_year)

    # --- ルックアップSeries構築（ハッシュベース高速マッピング） ---
    sire_ei_lu = _leading_lookup_series(sire_ld, "ei")
    bms_ei_lu = _leading_lookup_series(bms_ld, "ei")
    sire_wr_lu = _leading_lookup_series(sire_ld, "win_rate")
    bms_wr_lu = _leading_lookup_series(bms_ld, "win_rate")
    sire_runners_lu = _leading_lookup_series_int(sire_ld, "runners")
    sire_pp_lu = _leading_lookup_series(sire_ld, "progeny_prize")
    bms_runners_lu = _leading_lookup_series_int(bms_ld, "runners")
    sire_rank_lu = _leading_lookup_series_int(sire_ld, "rank")
    bms_rank_lu = _leading_lookup_series_int(bms_ld, "rank")
    bms_pp_lu = _leading_lookup_series(bms_ld, "progeny_prize")
    prev_sire_ei_lu = _leading_lookup_series(prev_sire_ld, "ei")

    # ソースカラム
    sire = src["sire"].fillna("")
    bms_name = src["sire_of_dam"].fillna("")
    dam = src["dam"].fillna("")
    hids = src["horse_id"].astype(str)

    # --- 結果DataFrame構築 ---
    r = pd.DataFrame(index=src.index)
    r["horse_id"] = src["horse_id"].values
    r["horse_name"] = src.get("horse_name", pd.Series("", index=src.index)).values

    # === 産駒成績ベースの血統スコア（ベクトル化マッピング） ===
    r["sire_ei"] = sire.map(sire_ei_lu).fillna(0.0)
    r["bms_ei"] = bms_name.map(bms_ei_lu).fillna(0.0)
    r["sire_win_rate"] = sire.map(sire_wr_lu).fillna(0.0)
    r["bms_win_rate"] = bms_name.map(bms_wr_lu).fillna(0.0)
    r["sire_runners"] = sire.map(sire_runners_lu).fillna(0).astype(int)
    r["sire_progeny_prize"] = sire.map(sire_pp_lu).fillna(0.0)
    r["bms_runners"] = bms_name.map(bms_runners_lu).fillna(0).astype(int)
    r["bms_progeny_prize"] = bms_name.map(bms_pp_lu).fillna(0.0)

    # ランク → 逆転スコア（1位=100, 50位≈2, 50位超=0）
    sire_rank_raw = sire.map(sire_rank_lu).fillna(0).astype(int)
    r["sire_rank_score"] = np.where(sire_rank_raw > 0, np.maximum(0, (51 - sire_rank_raw) * 2), 0.0)
    bms_rank_raw = bms_name.map(bms_rank_lu).fillna(0).astype(int)
    r["bms_rank_score"] = np.where(bms_rank_raw > 0, np.maximum(0, (51 - bms_rank_raw) * 2), 0.0)

    # 母馬賞金・種牡馬自身の賞金（辞書マッピング）
    dam_prizes_s = pd.Series(DAM_PRIZES)
    r["dam_prize"] = dam.map(dam_prizes_s).fillna(0.0)
    sire_prizes_s = pd.Series(SIRE_PRIZES)
    r["sire_prize"] = sire.map(sire_prizes_s).fillna(0.0)

    # === 調教師・馬主・生産者スコア ===
    r["trainer_score"] = src["trainer"].fillna("").apply(calc_trainer_score)
    r["owner_score"] = src["owner"].fillna("").apply(calc_owner_score)
    r["breeder_score"] = src["breeder"].fillna("").apply(calc_breeder_score)

    # === 生まれ月（ベクトル化） ===
    def _parse_birth_month(hid):
        bd = bd_cache.get(str(hid), "")
        if bd:
            m = re.search(r"(\d+)月", bd)
            if m:
                return int(m.group(1))
        return 3
    r["birth_month"] = hids.map(_parse_birth_month)

    # === 親年齢（CSVカラムから計算） ===
    def _calc_parent_ages_vec(parent_key):
        csv_col = _PARENT_MAP[parent_key][0]
        ages = pd.Series(np.nan, index=src.index)
        if csv_col in src.columns:
            parent_by = pd.to_numeric(src[csv_col], errors="coerce")
            valid = parent_by.notna()
            ages[valid] = birth_year - parent_by[valid]
        return ages

    r["sire_age"] = _calc_parent_ages_vec("sire")
    r["dam_age"] = _calc_parent_ages_vec("dam")
    r["dam_sire_age"] = _calc_parent_ages_vec("bms")

    # === バイナリ特徴量（ベクトル化） ===
    r["early_born"] = (r["birth_month"] <= 4).astype(int)
    r["sire_young"] = np.where(r["sire_age"].notna(), (r["sire_age"] <= 13).astype(float), np.nan)
    r["dam_young"] = np.where(r["dam_age"].notna(), (r["dam_age"] <= 13).astype(float), np.nan)
    both_notna = r["sire_age"].notna() & r["dam_age"].notna()
    r["both_parents_young"] = np.where(
        both_notna, ((r["sire_age"] <= 13) & (r["dam_age"] <= 13)).astype(float), np.nan
    )
    r["sire_first_crop"] = np.where(
        r["sire_age"].notna(), (r["sire_age"] <= 7).astype(float), np.nan
    )
    ds_notna = r["dam_sire_age"].notna() & r["dam_age"].notna()
    r["dam_bms_gap_small"] = np.where(
        ds_notna, ((r["dam_sire_age"] - r["dam_age"]) <= 15).astype(float), np.nan
    )

    # === セリ価格・産駒番号 ===
    def _get_sale_price(hid):
        extra = ex_cache.get(str(hid), {})
        sp = extra.get("sale_price")
        return np.log1p(sp) if sp else 0.0
    r["sale_price_log"] = hids.map(_get_sale_price)

    # 産駒番号: extra_features → dam_foals フォールバック
    dam_id_col = src["dam_id"].astype(str).where(src["dam_id"].notna(), "")

    def _get_foal_number(idx):
        hid_str = str(src.at[idx, "horse_id"])
        extra = ex_cache.get(hid_str, {})
        fn = extra.get("foal_number")
        if fn:
            return fn
        did = dam_id_col.at[idx]
        if did:
            fl = DAM_FOALS.get(did, [])
            if fl and hid_str in fl:
                return len(fl) - fl.index(hid_str)
        return 3.0
    r["foal_number"] = pd.Series(
        [_get_foal_number(idx) for idx in src.index], index=src.index
    )

    # === 母馬関連の特徴量（foal_listベース） ===
    classic_cutoff = birth_year - 2
    classic_ids = _get_classic_ids_up_to(classic_cutoff)

    def _dam_foal_features(idx):
        """母馬の繁殖入り年齢・産駒数・兄姉クラシック・産駒品質を一括計算。"""
        dam_by = src.at[idx, "dam_birth_year"] if "dam_birth_year" in src.columns else np.nan
        did = dam_id_col.at[idx]
        foal_list = DAM_FOALS.get(did, []) if did else []
        hid_str = str(src.at[idx, "horse_id"])

        # 繁殖入り年齢
        dam_breeding_age = np.nan
        if did and pd.notna(dam_by) and foal_list:
            first_foal_by = _birth_year_from_id(foal_list[-1])
            if first_foal_by is not None:
                dam_breeding_age = first_foal_by - int(dam_by)

        # 総産駒数（リーク防止）
        if foal_list:
            total_dam_foals = sum(1 for f in foal_list
                                  if (_birth_year_from_id(f) or 9999) <= birth_year)
        else:
            total_dam_foals = 0

        # 兄姉クラシック・産駒品質
        older_siblings = [f for f in foal_list
                          if (_birth_year_from_id(f) or 9999) < birth_year]
        sibling_classic = 1 if any(str(s) in classic_ids for s in older_siblings) else 0

        if older_siblings:
            classic_sibs = sum(1 for s in older_siblings if str(s) in classic_ids)
            dam_progeny_quality = classic_sibs / len(older_siblings)
        else:
            dam_progeny_quality = 0.0

        return dam_breeding_age, total_dam_foals, sibling_classic, dam_progeny_quality

    foal_results = [_dam_foal_features(idx) for idx in src.index]
    r["dam_breeding_age"] = [x[0] for x in foal_results]
    r["total_dam_foals"] = [x[1] for x in foal_results]
    r["sibling_classic"] = [x[2] for x in foal_results]
    r["dam_progeny_quality"] = [x[3] for x in foal_results]

    # === クラシック輩出数（ベクトル化マッピング） ===
    sire_classic_map = _get_sire_classic_map(classic_cutoff)
    sire_oaks_map = _get_sire_oaks_map(classic_cutoff)
    bms_classic_map = _get_bms_classic_map(classic_cutoff)

    sire_classic_lu = pd.Series(sire_classic_map)
    sire_oaks_lu = pd.Series(sire_oaks_map)
    bms_classic_lu = pd.Series(bms_classic_map)

    r["sire_classic_count"] = sire.map(sire_classic_lu).fillna(0).astype(int)
    sire_runners_v = r["sire_runners"]
    r["sire_classic_rate"] = np.where(sire_runners_v > 0,
                                      r["sire_classic_count"] / sire_runners_v * 100, 0.0)

    r["sire_oaks_count"] = sire.map(sire_oaks_lu).fillna(0).astype(int)
    r["sire_oaks_rate"] = np.where(sire_runners_v > 0,
                                   r["sire_oaks_count"] / sire_runners_v * 100, 0.0)

    # 種牡馬ダービー率
    sire_derby_map = _get_sire_derby_map(classic_cutoff)
    sire_derby_lu = pd.Series(sire_derby_map)
    r["sire_derby_count"] = sire.map(sire_derby_lu).fillna(0).astype(int)
    r["sire_derby_rate"] = np.where(sire_runners_v > 0,
                                    r["sire_derby_count"] / sire_runners_v * 100, 0.0)

    r["bms_classic_count"] = bms_name.map(bms_classic_lu).fillna(0).astype(int)
    bms_runners_v = r["bms_runners"]

    # 母父オークス率
    bms_oaks_map = _get_bms_oaks_map(classic_cutoff)
    bms_oaks_lu = pd.Series(bms_oaks_map)
    r["bms_oaks_count"] = bms_name.map(bms_oaks_lu).fillna(0).astype(int)
    r["bms_oaks_rate"] = np.where(bms_runners_v > 0,
                                  r["bms_oaks_count"] / bms_runners_v * 100, 0.0)

    # 母父ダービー率
    bms_derby_map = _get_bms_derby_map(classic_cutoff)
    bms_derby_lu = pd.Series(bms_derby_map)
    r["bms_derby_count"] = bms_name.map(bms_derby_lu).fillna(0).astype(int)
    r["bms_derby_rate"] = np.where(bms_runners_v > 0,
                                   r["bms_derby_count"] / bms_runners_v * 100, 0.0)

    # 母馬クラシック実績（母馬がオークスTOP5だったか）
    dam_classic_ids = _get_dam_classic_ids(classic_cutoff)
    r["dam_classic"] = dam_id_col.isin(dam_classic_ids).astype(int)

    # 種牡馬平均賞金（産駒あたり賞金 = 質の指標）
    r["sire_mean_prize"] = np.where(sire_runners_v > 0,
                                    r["sire_progeny_prize"] / sire_runners_v, 0.0)

    # === 派生特徴量（全ベクトル化） ===
    r["dam_high_class"] = (r["dam_prize"] >= 5000).astype(int)

    # EIトレンド（前年比）
    prev_ei = sire.map(prev_sire_ei_lu).fillna(0.0)
    r["sire_ei_trend"] = np.where(prev_ei > 0, r["sire_ei"] - prev_ei, 0.0)

    # 輸入繁殖牝馬フラグ
    r["imported_dam"] = dam_id_col.str.startswith("000a").astype(int)

    # 輸入繁殖牝馬の評価補正（外国産母馬はdam_prize=0, bms_ei≈0で減点される問題を緩和）
    r["imported_sire_inter"] = r["imported_dam"] * r["sire_ei"]
    r["imported_trainer_inter"] = r["imported_dam"] * (r["trainer_score"] - 50) / 50
    r["imported_owner_inter"] = r["imported_dam"] * (r["owner_score"] - 50) / 50
    # 外国産母馬 × 母父品質（dam_prizeが0でも母父の質で母系を評価）
    r["imported_bms_inter"] = r["imported_dam"] * r["bms_ei"]
    r["imported_bms_rank_inter"] = r["imported_dam"] * r["bms_rank_score"] / 100
    r["imported_breeder_inter"] = r["imported_dam"] * (r["breeder_score"] - 50) / 50

    # === 交互作用特徴量（ベクトル化） ===
    r["sire_dam_interaction"] = r["sire_ei"] * np.log1p(r["dam_prize"])
    r["trainer_breeder_combo"] = r["trainer_score"] * r["breeder_score"]
    r["owner_trainer_combo"] = r["owner_score"] * r["trainer_score"]
    r["bms_dam_interaction"] = r["bms_ei"] * np.log1p(r["dam_prize"])

    return r.reset_index(drop=True)
