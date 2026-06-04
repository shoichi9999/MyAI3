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


# === 血統理論：父Stayer × 母父Miler フィルタ ===
# Why: 日本ダービー(2400m)で「父=ステイヤー型(中長距離G1実績)」かつ「母父=マイラー型(マイル〜中距離スピード)」の
# 配合が好走しやすいというPOG/血統理論。「父スタミナ×母系スピード」の古典的バランス。
STAYERS_SET = frozenset([
    # 本格派ステイヤー（2800m以上G1勝ち、または産駒に3200m級ステイヤーG1馬を多数輩出）
    "キタサンブラック",     # 菊3000・天春3200
    "ゴールドシップ",       # 天春3200・菊3000
    "ディープインパクト",   # 菊3000・天春3200・JC2400・有馬2500
    "エピファネイア",       # 菊3000・JC2400
    "ワールドプレミア",     # 菊3000・天春3200
    "マンハッタンカフェ",   # 菊3000・天春3200・有馬2500
    "オルフェーヴル",       # 菊3000・凱旋門賞2着
    "タイトルホルダー",     # 菊3000・天春3200
    "フィエールマン",       # 菊3000・天春3200
    "コントレイル",         # 菊3000・三冠馬
    "メイショウサムソン",   # 天春3200
    "オウケンブルースリ",   # 菊3000
    "Monsun",               # 産駒にメルボルンC(3200m)勝ち3頭、ドイツダービー馬輩出
])
MILERS_SET = frozenset([
    # 国内
    "ダイワメジャー", "ロードカナロア", "アドマイヤマーズ", "モーリス", "クロフネ",
    "フジキセキ", "アグネスタキオン", "ダンスインザダーク", "サクラバクシンオー",
    "グラスワンダー", "ブラックタイド", "アグネスデジタル", "ノヴェリスト",
    "キンシャサノキセキ", "ミッキーアイル", "キングカメハメハ", "キングマンボ",
    "シンボリクリスエス", "タイキシャトル", "フレンチデピュティ", "ブライアンズタイム",
    "ハービンジャー", "リアルスティール", "リアルインパクト", "スクリーンヒーロー",
    "メイショウボーラー", "ヘニーヒューズ", "ジャングルポケット",
    # 海外マイル/中距離G1系（Galileo, Sea The Stars, Monsun は STAYERS_SET に移動）
    "Frankel", "Dubawi", "War Front", "Tapit", "Curlin", "Speightstown",
    "Hard Spun", "Smart Strike", "Distorted Humor", "Storm Cat", "Kingmambo",
    "Mr. Prospector", "Nureyev", "Northern Dancer", "Sadler's Wells", "Danehill",
    "More Than Ready", "Medaglia d'Oro", "Bernardini", "Pulpit", "Lemon Drop Kid",
    "Giant's Causeway", "Awesome Again", "A.P. Indy", "Fastnet Rock", "Snitzel",
    "Redoute's Choice", "Encosta de Lago", "Rock of Gibraltar", "Invincible Spirit",
    "Oasis Dream", "Pivotal", "Dansili", "Iffraaj", "Kodiac", "Lope De Vega",
    "No Nay Never", "New Approach", "Shamardal", "Singspiel",  # Sea The Stars は STAYERS へ移動
    "Exceed And Excel", "Laoban", "Street Cry", "Empire Maker",
    "Unbridled's Song", "Forestry", "Quality Road", "Into Mischief", "Uncle Mo",
    "Honor Code", "Constitution", "American Pharoah", "Justify",
    # オークス1着馬の母父（実証ベース追加）
    "サンデーサイレンス",   # アーモンドアイ母父
    "ロージズインメイ",     # ユーバーレーベン母父
    "All American",         # リバティアイランド母父
    # 過去ダービー1着馬の母父（実証ベース追加）
    "Librettist", "Essence of Dubai", "Vindication", "Congrats", "Cape Cross",
    # 欧米マイル〜中距離G1馬（要追加）
    "Le Havre",     # 仏マイラー、産駒に仏オークス馬・仏ダービー牝馬
    "Wootton Bassett",  # 欧マイラーG1、産駒に欧G1多数
    "Intello",      # 仏ダービー馬、マイラー〜中距離
    # 加えて主要な欧米マイラー系種牡馬
    "Storm Bird", "Bel Esprit", "Tale of the Cat", "Distant View",
    "Seeking the Gold", "Gone West", "Carson City", "Mr. Greeley",
    "Roberto", "Hail to Reason", "Bold Ruler", "Raise a Native",
    "Halo", "Lyphard", "Vice Regent", "Vaguely Noble",
    "Caerleon", "Sharpen Up", "Habitat", "Nasrullah",
    "Nashua", "Bold Bidder", "Northern Taste", "Caro",
    "Riverman", "Cox's Ridge", "Topsider", "Storm Bird",
    "Rahy", "Singletary", "Pleasant Tap", "Maria's Mon",
    "Officer", "Stravinsky", "Rock Hard Ten", "Tiznow",
    "Awesome Again", "Macho Uno", "Hennessy", "Old Trieste",
    "Coronado's Quest", "Petionville", "Mt. Livermore",
    "Spend a Buck", "Slew o' Gold", "Capote", "Easy Goer",
    "Phone Trick", "Dixieland Band", "Saint Ballado", "Salt Lake",
])

# 中距離G1馬（ダービー2400m/オークス2400mで産駒実績ある中距離G1勝ち種牡馬）
# Why: オークスでは父スタミナ要件が緩く、中距離型の父でも勝てる。
# 「父中距離以上 × 母父スピード」のオークス理論の左辺に使用。
MID_DIST_SET = frozenset([
    "ドゥラメンテ",         # 皐月・ダービー
    "ハービンジャー",       # キングジョージ&クイーンエリザベス（欧2400m G1）
    "ジャスタウェイ",       # 天秋・安田・ドバイDF
    "ルーラーシップ",       # キングジョージ&クイーンエリザベスC
    "レイデオロ",           # ダービー・天秋
    "サートゥルナーリア",   # 皐月・ホープフル
    "スワーヴリチャード",   # 大阪杯・JC
    "ヴィクトワールピサ",   # 皐月・有馬・ドバイWC
    "ネオユニヴァース",     # 皐月・ダービー
    "ダノンキングリー",     # 中距離G1
    "シュヴァルグラン",     # JC
    "ステイゴールド",       # 香港V2400・有馬2500
    "ジャングルポケット",   # JC・ダービー
    "キズナ",               # ダービー・産経大阪杯・凱旋門賞4着
    # 本格派ステイヤーから降格（2400m帯中長距離型）
    "ハーツクライ",         # 有馬2500・宝塚2200・ドバイシーマ2400
    "ゴールドアクター",     # 有馬2500
    "ブラストワンピース",   # 有馬2500
    "ナカヤマフェスタ",     # 宝塚2200・凱旋門賞2着
    "イクイノックス",       # JC2400・有馬2500・天秋2000・ドバイSC
    "Galileo",              # 英ダービー2400・キングジョージ2406
    "Sea The Stars",        # 凱旋門賞2400・英ダービー2400
])

# スプリンター系種牡馬（短距離G1勝ち、もしくは短距離血統として広く認知される）
SPRINTERS_SET = frozenset([
    # 国内
    "サクラバクシンオー",   # スプリンターズS連覇
    "ロードカナロア",       # 高松宮・スプリンターズ・香港スプリント
    "ダノンスマッシュ",     # 高松宮・香港スプリント
    "レッドファルクス",     # スプリンターズS連覇
    "ダノンレジェンド",     # 東京盃
    "マイネルラヴ",         # スプリンターズS
    "アドマイヤコジーン",   # 安田・短距離G1
    "ヘンリーバローズ",
    "スウェプトオーヴァーボード",  # スプリンターズS
    "プリサイスエンド",
    # 海外スプリンター
    "Danzig", "Storm Cat", "Mr. Greeley", "Tale of the Cat",
    "Forestry", "Speightstown", "Spinning World", "Distorted Humor",
    "More Than Ready",
    "Street Boss",  # 米スプリンターG1、父Street Cry
    "Sightseeing",  # 米マイラー寄り
])

# デフォルト（最新）のリーディングデータ — 予測時に使用
_DEFAULT_LEADING_YEAR = 2024
SIRE_LEADING = _load_json("data/sire_leading_2024.json")

# 種牡馬の距離適性（自身の競走時G1〜G3勝ち距離から自動判定）
# horse_id -> "STAYER"/"MID_DIST"/"MILER"/"SPRINTER" or None
_SIRE_DISTANCE = _load_json("data/sire_distance.json")

# 種牡馬の累計成績データ（重賞勝ち数・平均距離・EI、父用と母父用の両方）
_SIRE_STATS = _load_json("data/sire_stats.json")


def get_distance_category(horse_id: str) -> str | None:
    """horse_idから距離適性カテゴリを返す。データなしなら None。"""
    if not horse_id:
        return None
    data = _SIRE_DISTANCE.get(str(horse_id))
    if not data or not data.get("wins"):
        return None
    mx = max(data["wins"])
    if mx >= 2800:
        return "STAYER"
    elif mx >= 1800:
        return "MID_DIST"
    elif mx >= 1400:
        return "MILER"
    else:
        return "SPRINTER"


def get_sire_graded(horse_id: str, role: str = "sire") -> int:
    """種牡馬の累計重賞勝ち数。role='sire'/'bms'。データなしは 0。"""
    if not horse_id:
        return 0
    data = _SIRE_STATS.get(str(horse_id))
    if not data or not data.get(role):
        return 0
    return int(data[role].get("graded", 0))


def get_sire_avg_dist_turf(horse_id: str, role: str = "sire") -> float:
    """種牡馬の平均距離（芝）。データなしは 0。"""
    if not horse_id:
        return 0.0
    data = _SIRE_STATS.get(str(horse_id))
    if not data or not data.get(role):
        return 0.0
    return float(data[role].get("avg_dist_turf", 0))
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
    sire_2yo_ld = _load_leading("sire_2yo_leading", leading_year)
    prev_sire_ld = _load_leading("sire_leading", leading_year - 1)
    bd_cache = _load_birth_dates(birth_year)
    pa_cache = _load_parent_ages(birth_year)
    ex_cache = _load_extra_features(birth_year)

    # --- ルックアップSeries構築（ハッシュベース高速マッピング） ---
    sire_ei_lu = _leading_lookup_series(sire_ld, "ei")
    bms_ei_lu = _leading_lookup_series(bms_ld, "ei")
    sire_2yo_ei_lu = _leading_lookup_series(sire_2yo_ld, "ei")
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
    r["sire_2yo_ei"] = sire.map(sire_2yo_ei_lu).fillna(0.0)
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

    # === 親年齢（CSV優先 → JSONキャッシュフォールバック） ===
    def _calc_parent_ages_vec(parent_key):
        csv_col, cache_by_key, cache_id_key = _PARENT_MAP.get(
            parent_key, (f"{parent_key}_birth_year", f"{parent_key}_birth_year", None)
        )
        # 1. CSVカラムから計算
        ages = pd.Series(np.nan, index=src.index)
        if csv_col in src.columns:
            parent_by = pd.to_numeric(src[csv_col], errors="coerce")
            valid = parent_by.notna()
            ages[valid] = birth_year - parent_by[valid]
        # 2. JSONキャッシュからフォールバック（NaN残りのみ）
        missing = ages.isna()
        if missing.any():
            for idx in src.index[missing]:
                hid_str = str(src.at[idx, "horse_id"])
                data = pa_cache.get(hid_str, {})
                by = data.get(cache_by_key)
                if by:
                    ages.at[idx] = birth_year - by
                elif cache_id_key:
                    parent_id = data.get(cache_id_key)
                    if parent_id:
                        parent_by_val = _birth_year_from_id(parent_id)
                        if parent_by_val:
                            ages.at[idx] = birth_year - parent_by_val
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
    # 信頼度補正: 産駒数 < RATE_MIN_RUNNERS の種牡馬は率を 0（情報なし扱い）にする。
    # Why: Siyouni等の少産駒種牡馬で「1/4=25%」を真の能力と見なすのも、
    #      全体平均に引き戻すのも、どちらも妥当性に欠ける。"わからない" を素直に
    #      中立(0)として扱い、他項目（父EI・ランク・配合理論）で評価する。
    RATE_MIN_RUNNERS = 20
    r["sire_classic_rate"] = np.where(sire_runners_v >= RATE_MIN_RUNNERS,
                                      r["sire_classic_count"] / sire_runners_v * 100, 0.0)

    r["sire_oaks_count"] = sire.map(sire_oaks_lu).fillna(0).astype(int)
    r["sire_oaks_rate"] = np.where(sire_runners_v >= RATE_MIN_RUNNERS,
                                   r["sire_oaks_count"] / sire_runners_v * 100, 0.0)

    r["bms_classic_count"] = bms_name.map(bms_classic_lu).fillna(0).astype(int)

    # === 派生特徴量（全ベクトル化） ===
    r["dam_high_class"] = (r["dam_prize"] >= 5000).astype(int)
    r["sire_precocity"] = np.where(r["sire_ei"] > 0, r["sire_2yo_ei"] / r["sire_ei"], 0.0)

    # EIトレンド（前年比）
    prev_ei = sire.map(prev_sire_ei_lu).fillna(0.0)
    r["sire_ei_trend"] = np.where(prev_ei > 0, r["sire_ei"] - prev_ei, 0.0)

    # 輸入繁殖牝馬フラグ
    r["imported_dam"] = dam_id_col.str.startswith("000a").astype(int)
    # 産地別ボーナス（過去9年TOP5実績比率から決定）
    # Why: 一律加点だと米国産（全体49%、TOP5率35%＝倍率0.71）を過大評価し、
    # 独・亜（倍率4.15/3.13）を過小評価する。実態に合わせて補正。
    _COUNTRY_BONUS = {
        "独": 15, "亜": 12, "豪": 6, "英": 3,
        "米": 0, "愛": 0, "仏": 0,
    }
    _country_map = _load_json("data/dam_countries.json")
    _country_lu = dam_id_col.map(_country_map)
    r["dam_country_bonus"] = _country_lu.map(_COUNTRY_BONUS).fillna(0.0).astype(float)

    # === 交互作用特徴量（ベクトル化） ===
    r["sire_dam_interaction"] = r["sire_ei"] * np.log1p(r["dam_prize"])
    r["trainer_breeder_combo"] = r["trainer_score"] * r["breeder_score"]
    r["owner_trainer_combo"] = r["owner_score"] * r["trainer_score"]
    r["bms_dam_interaction"] = r["bms_ei"] * np.log1p(r["dam_prize"])

    # === 父Stayer × 母父Miler 配合フラグ（血統理論ベース） ===
    # 手動リスト + 競走時距離適性データ(sire_distance.json) のいずれかでマッチ
    sire_id_col = src["sire_id"].astype(str).fillna("")
    bms_id_col = src["bms_id"].astype(str).fillna("") if "bms_id" in src.columns else pd.Series("", index=src.index)
    sire_dist_cat = sire_id_col.apply(get_distance_category)
    bms_dist_cat = bms_id_col.apply(get_distance_category)
    r["sire_is_stayer"] = (sire.isin(STAYERS_SET) | (sire_dist_cat == "STAYER")).astype(int).values
    r["sire_is_mid_dist"] = (sire.isin(MID_DIST_SET) | (sire_dist_cat == "MID_DIST")).astype(int).values
    r["bms_is_miler"] = (bms_name.isin(MILERS_SET) | (bms_dist_cat == "MILER")).astype(int).values
    r["bms_is_sprinter"] = (bms_name.isin(SPRINTERS_SET) | (bms_dist_cat == "SPRINTER")).astype(int).values

    # ダービー用: 父Stayer × 母父Miler
    r["stayer_x_miler"] = (r["sire_is_stayer"] & r["bms_is_miler"]).astype(int)
    # オークス用: 父中距離以上 (Stayer+MidDist) × 母父スピード (Miler+Sprinter)
    sire_mid_or_up = (r["sire_is_stayer"] | r["sire_is_mid_dist"]).astype(int)
    bms_speed = (r["bms_is_miler"] | r["bms_is_sprinter"]).astype(int)
    r["bms_is_speed"] = bms_speed
    r["mid_x_speed"] = (sire_mid_or_up & bms_speed).astype(int)

    return r.reset_index(drop=True)
