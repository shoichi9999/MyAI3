"""
2023年生まれ世代のバックテスト。
血統（先祖の実賞金 × 世代係数）で予測し、実際の賞金ランキングと比較。
"""
import time
import re
import json
import os
import urllib.parse

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from src.scraper import _get_soup, BASE_URL, REQUEST_INTERVAL


CACHE_FILE = "data/ancestor_prizes.json"


def parse_prize_text(text: str) -> float:
    """賞金テキスト(例: '14億5,455万円')を万円単位のfloatに変換。"""
    if not text:
        return 0.0
    text = text.replace(" ", "").replace(",", "").replace("円", "")
    total = 0.0
    # 億の処理
    m = re.search(r"(\d+)億", text)
    if m:
        total += int(m.group(1)) * 10000
    # 万の処理
    m = re.search(r"(\d+)万", text)
    if m:
        total += int(m.group(1))
    return total


def fetch_horse_prize_by_name(name: str) -> float:
    """
    netkeiba の名前検索で馬の獲得賞金(万円)を取得する。
    プロフィールページへの直接リダイレクト、一覧表示の両方に対応。
    """
    if not name or name.strip() == "":
        return 0.0

    encoded = urllib.parse.quote(name.strip())
    url = f"{BASE_URL}/?pid=horse_list&word={encoded}&sort=prize&list=100"

    try:
        soup = _get_soup(url)
    except Exception as e:
        print(f"  [WARN] {name} の検索失敗: {e}")
        return 0.0

    # パターン1: プロフィールページに直接遷移
    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table:
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "賞金" in th.text and "中央" in th.text:
                return parse_prize_text(td.text)
        # 中央がなければ地方も
        for row in prof_table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "賞金" in th.text:
                return parse_prize_text(td.text)

    # パターン2: 一覧テーブル表示
    list_table = soup.find("table", class_="nk_tb_common")
    if list_table:
        rows = list_table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 12:
                horse_name = cols[1].text.strip()
                if horse_name == name:
                    prize_text = cols[11].text.strip()
                    return float(prize_text.replace(",", "")) if prize_text else 0.0

    return 0.0


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def collect_ancestor_prizes(horses_df: pd.DataFrame) -> dict:
    """
    ユニークな父馬・母父馬の賞金を一括取得する。
    キャッシュ対応。
    """
    cache = load_cache()

    # ユニークな先祖名（父 + 母父）
    sires = set(horses_df["sire"].dropna().unique())
    bms_set = set(horses_df["sire_of_dam"].dropna().unique())
    all_ancestors = sires | bms_set
    # 空文字除去
    all_ancestors = {a for a in all_ancestors if a.strip()}

    # キャッシュにないものだけ取得
    to_fetch = [a for a in all_ancestors if a not in cache]
    print(f"先祖馬: {len(all_ancestors)}頭 (キャッシュ済: {len(all_ancestors)-len(to_fetch)}, 新規: {len(to_fetch)})")

    for i, name in enumerate(to_fetch):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(to_fetch)}] {name} ...")
        prize = fetch_horse_prize_by_name(name)
        cache[name] = prize

        # 50件ごとにキャッシュ保存
        if (i + 1) % 50 == 0:
            save_cache(cache)

    save_cache(cache)
    print(f"全先祖の賞金取得完了")
    return cache


def calc_bloodline_score(row, prize_cache: dict) -> float:
    """
    世代係数ベースの血統スコアを計算する。
    
    score = sire_prize * 0.4 + bms_prize * 0.25 + (残りは調教師等で別途)
    ※ 賞金は対数スケールで正規化
    """
    sire = row.get("sire", "")
    bms = row.get("sire_of_dam", "")

    sire_prize = prize_cache.get(sire, 0) if sire else 0
    bms_prize = prize_cache.get(bms, 0) if bms else 0

    return sire_prize, bms_prize


def main():
    # 2023年生まれのデータ読み込み
    horses = pd.read_csv("data/horses_2023.csv")
    horses["prize_num"] = horses["total_prize"].apply(
        lambda x: float(str(x).replace(",", "").replace("万", "").strip())
        if pd.notna(x) and str(x).strip() else 0.0
    )

    print("=" * 70)
    print("  先祖の実賞金 × 世代係数 による血統スコアモデル")
    print("=" * 70)

    # 1. 先祖の賞金を取得
    prize_cache = collect_ancestor_prizes(horses)

    # 2. 各馬の血統スコアを計算
    sire_prizes = []
    bms_prizes = []
    for _, row in horses.iterrows():
        sp, bp = calc_bloodline_score(row, prize_cache)
        sire_prizes.append(sp)
        bms_prizes.append(bp)

    horses["sire_prize"] = sire_prizes
    horses["bms_prize"] = bms_prizes

    # 対数スケールで正規化 (0-100)
    horses["sire_log"] = np.log1p(horses["sire_prize"])
    horses["bms_log"] = np.log1p(horses["bms_prize"])

    sire_max = horses["sire_log"].max()
    bms_max = horses["bms_log"].max()
    if sire_max > 0:
        horses["sire_norm"] = horses["sire_log"] / sire_max * 100
    else:
        horses["sire_norm"] = 50
    if bms_max > 0:
        horses["bms_norm"] = horses["bms_log"] / bms_max * 100
    else:
        horses["bms_norm"] = 50

    # 調教師スコア (既存の辞書を使用)
    from src.features import calc_trainer_score
    horses["trainer_score"] = horses["trainer"].apply(
        lambda x: calc_trainer_score(str(x)) if pd.notna(x) else 50
    )

    # 世代係数ベースの総合スコア
    # 父(gen1): 0.45, 母父(gen2): 0.25, 調教師: 0.30
    horses["bloodline_score"] = (
        horses["sire_norm"] * 0.45 +
        horses["bms_norm"] * 0.25 +
        horses["trainer_score"] * 0.30
    )

    # 3. 予測 vs 実績の比較
    pred_top = horses.nlargest(30, "bloodline_score")
    actual_top = horses.nlargest(30, "prize_num")

    pred_top30_ids = set(pred_top["horse_id"].values)
    actual_top30_ids = set(actual_top["horse_id"].values)

    pred_top10 = horses.nlargest(10, "bloodline_score")
    actual_top10 = horses.nlargest(10, "prize_num")
    pred_top10_ids = set(pred_top10["horse_id"].values)
    actual_top10_ids = set(actual_top10["horse_id"].values)

    pred_top50 = horses.nlargest(50, "bloodline_score")
    actual_top50 = horses.nlargest(50, "prize_num")
    pred_top50_ids = set(pred_top50["horse_id"].values)
    actual_top50_ids = set(actual_top50["horse_id"].values)

    # 表示
    print("\n--- 新モデル予測 TOP20 ---")
    print(f"{'順位':>4s} {'馬名':<20s} {'スコア':>8s} {'父賞金':>10s} {'母父賞金':>10s} | {'実賞金':>10s} {'実順位':>6s}")
    print("-" * 85)
    for i, (_, row) in enumerate(pred_top.head(20).iterrows(), 1):
        actual_rank = int(horses["prize_num"].rank(ascending=False)[horses["horse_id"] == row["horse_id"]].values[0])
        mark = "★" if actual_rank <= 30 else "  "
        print(f"{i:>4d} {row['horse_name']:<20s} {row['bloodline_score']:>8.1f} "
              f"{row['sire_prize']:>10,.0f} {row['bms_prize']:>10,.0f} | "
              f"{row['prize_num']:>8,.0f}万 {actual_rank:>4d}位 {mark}")

    print("\n--- 実際の賞金 TOP20 ---")
    print(f"{'順位':>4s} {'馬名':<20s} {'実賞金':>10s} | {'予測スコア':>8s} {'予測順位':>6s}")
    print("-" * 65)
    for i, (_, row) in enumerate(actual_top.head(20).iterrows(), 1):
        pred_rank = int(horses["bloodline_score"].rank(ascending=False)[horses["horse_id"] == row["horse_id"]].values[0])
        mark = "★" if pred_rank <= 30 else "  "
        print(f"{i:>4d} {row['horse_name']:<20s} {row['prize_num']:>8,.0f}万 | "
              f"{row['bloodline_score']:>8.1f} {pred_rank:>4d}位 {mark}")

    # 精度指標
    overlap_10 = len(pred_top10_ids & actual_top10_ids)
    overlap_30 = len(pred_top30_ids & actual_top30_ids)
    overlap_50 = len(pred_top50_ids & actual_top50_ids)

    pred_top30_avg = horses.loc[horses["horse_id"].isin(pred_top30_ids), "prize_num"].mean()
    overall_avg = horses["prize_num"].mean()
    actual_top30_avg = actual_top["prize_num"].mean()

    corr, pval = spearmanr(horses["bloodline_score"], horses["prize_num"])

    print("\n" + "=" * 70)
    print("  検証結果サマリー（新モデル: 先祖実賞金 × 世代係数）")
    print("=" * 70)
    print(f"  TOP10 一致数: {overlap_10}/10  ({overlap_10 * 10}%)")
    print(f"  TOP30 一致数: {overlap_30}/30  ({overlap_30 / 30 * 100:.1f}%)")
    print(f"  TOP50 一致数: {overlap_50}/50  ({overlap_50 / 50 * 100:.1f}%)")
    print(f"")
    print(f"  全馬平均賞金:           {overall_avg:>10,.0f}万")
    print(f"  予測TOP30の実際の平均賞金: {pred_top30_avg:>10,.0f}万")
    print(f"  実際TOP30の平均賞金:     {actual_top30_avg:>10,.0f}万")
    print(f"  予測TOP30の賞金倍率:     {pred_top30_avg / overall_avg:.1f}x (全体平均比)")
    print(f"")
    print(f"  Spearman順位相関:       {corr:.4f} (p={pval:.2e})")
    print("=" * 70)

    # 新モデル（EI + 母馬賞金）との比較
    from src.features import get_sire_ei, get_bms_ei, get_dam_prize
    from src.features import WEIGHT_SIRE_EI, WEIGHT_DAM_PRIZE, WEIGHT_BMS_EI, WEIGHT_TRAINER
    from src.model import POGPredictor

    horses["sire_ei"] = horses["sire"].apply(lambda x: get_sire_ei(str(x)) if pd.notna(x) else 0)
    horses["bms_ei"] = horses["sire_of_dam"].apply(lambda x: get_bms_ei(str(x)) if pd.notna(x) else 0)
    horses["dam_prize_val"] = horses["dam"].apply(lambda x: get_dam_prize(str(x)) if pd.notna(x) else 0)

    # EIの正規化
    sire_ei_max = horses["sire_ei"].max()
    bms_ei_max = horses["bms_ei"].max()
    dam_log = np.log1p(horses["dam_prize_val"])
    dam_log_max = dam_log.max()

    horses["ei_score"] = (
        (horses["sire_ei"] / sire_ei_max * 100 if sire_ei_max > 0 else 0) * WEIGHT_SIRE_EI +
        (horses["bms_ei"] / bms_ei_max * 100 if bms_ei_max > 0 else 0) * WEIGHT_BMS_EI +
        (dam_log / dam_log_max * 100 if dam_log_max > 0 else 0) * WEIGHT_DAM_PRIZE +
        horses["trainer_score"] * WEIGHT_TRAINER
    )

    ei_corr, _ = spearmanr(horses["ei_score"], horses["prize_num"])

    # EIモデルの TOP30 一致
    ei_top30 = horses.nlargest(30, "ei_score")
    ei_top30_ids = set(ei_top30["horse_id"].values)
    ei_overlap_30 = len(ei_top30_ids & actual_top30_ids)

    ei_top30_avg = horses.loc[horses["horse_id"].isin(ei_top30_ids), "prize_num"].mean()

    print(f"\n  [比較] 先祖賞金モデル Spearman: {corr:.4f},  TOP30一致: {overlap_30}/30")
    print(f"  [比較] EI＋母馬賞金モデル Spearman: {ei_corr:.4f},  TOP30一致: {ei_overlap_30}/30")
    print(f"  [比較] EIモデル TOP30平均賞金: {ei_top30_avg:>10,.0f}万 ({ei_top30_avg / overall_avg:.1f}x)")
    if abs(corr) > 0:
        print(f"  Spearman改善率: {(ei_corr - corr) / abs(corr) * 100:+.1f}%")


if __name__ == "__main__":
    main()
