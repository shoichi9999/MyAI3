"""
2015-2022年の全データを一括取得するスクリプト。

取得するもの:
1. horses_{year}.csv — 各年の馬リスト
2. sire_leading_{year+1}.json — リーディングサイアー
3. bms_leading_{year+1}.json — BMSリーディング
4. dam_prizes.json — 新規母馬の賞金データ追加
5. dam_foals.json — 新規母馬の産駒リスト追加
6. sire_prizes.json — 新規種牡馬の賞金データ追加
"""

import json
import os
import sys
import time

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scraper import fetch_horse_list_by_year, fetch_dam_foal_list, concurrent_fetch, _rate_limited_sleep, _get_soup
from fetch_leading import fetch_all_leading


TARGET_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def fetch_horse_lists():
    """不足している年度の馬リストを取得してCSVに保存する。"""
    print("\n" + "=" * 60)
    print("  Phase 1: 馬リスト取得")
    print("=" * 60)

    for year in TARGET_YEARS:
        csv_path = f"data/horses_{year}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"  {year}年: スキップ（既存 {len(df)}頭）")
            continue

        print(f"\n  {year}年: 取得開始...")
        df = fetch_horse_list_by_year(year)
        if not df.empty:
            df.to_csv(csv_path, index=False)
            print(f"  {year}年: {len(df)}頭を保存 → {csv_path}")
        else:
            print(f"  {year}年: [ERROR] データ取得失敗")


def fetch_leading_data():
    """不足しているリーディングデータを取得する。"""
    print("\n" + "=" * 60)
    print("  Phase 2: リーディングデータ取得")
    print("=" * 60)

    # 各年度に必要なリーディング年 = birth_year + 1
    needed_years = set()
    for y in TARGET_YEARS:
        needed_years.add(y + 1)

    for year in sorted(needed_years):
        for kind in ["sire_leading", "bms_leading"]:
            out_path = f"data/{kind}_{year}.json"
            if os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
                print(f"  {kind}_{year}: スキップ（既存 {len(existing)}件）")
                continue

            print(f"\n  {kind}_{year}: 取得開始...")
            data = fetch_all_leading(kind, year, max_pages=20)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  {kind}_{year}: {len(data)}件を保存")
            time.sleep(1.0)


def fetch_dam_data():
    """全年度の母馬データ（賞金・産駒リスト）を更新する。"""
    print("\n" + "=" * 60)
    print("  Phase 3: 母馬データ更新")
    print("=" * 60)

    # 既存データ読み込み
    dam_prizes_path = "data/dam_prizes.json"
    dam_foals_path = "data/dam_foals.json"

    if os.path.exists(dam_prizes_path):
        with open(dam_prizes_path, "r", encoding="utf-8") as f:
            dam_prizes = json.load(f)
    else:
        dam_prizes = {}

    if os.path.exists(dam_foals_path):
        with open(dam_foals_path, "r", encoding="utf-8") as f:
            dam_foals = json.load(f)
    else:
        dam_foals = {}

    # 全年度のCSVから母馬IDを収集
    all_dams = {}  # dam_id → dam_name
    for year in TARGET_YEARS:
        csv_path = f"data/horses_{year}.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            dam_id = str(row.get("dam_id", ""))
            dam_name = str(row.get("dam", ""))
            if dam_id and dam_name and dam_id != "nan":
                all_dams[dam_id] = dam_name

    print(f"  全年度のユニーク母馬: {len(all_dams)}頭")
    print(f"  既存賞金データ: {len(dam_prizes)}頭")
    print(f"  既存産駒リスト: {len(dam_foals)}頭")

    # 賞金データが未取得の母馬
    missing_prize_dams = {did: name for did, name in all_dams.items()
                          if name not in dam_prizes}
    # 産駒リストが未取得の母馬
    missing_foal_dams = [did for did in all_dams if did not in dam_foals]

    print(f"  賞金未取得: {len(missing_prize_dams)}頭")
    print(f"  産駒リスト未取得: {len(missing_foal_dams)}頭")

    # 母馬の賞金を取得
    if missing_prize_dams:
        print(f"\n  母馬賞金データ取得中...")

        def fetch_dam_prize(dam_id):
            """母馬の獲得賞金を取得する。"""
            _rate_limited_sleep()
            url = f"https://db.netkeiba.com/horse/{dam_id}/"
            try:
                soup = _get_soup(url)
                # 賞金テーブルを探す
                for table in soup.find_all("table"):
                    # プロフィールテーブルの「本賞金」行を探す
                    for row in table.find_all("tr"):
                        th = row.find("th")
                        td = row.find("td")
                        if th and td and "本賞金" in th.text:
                            text = td.text.strip()
                            try:
                                return float(text.replace(",", "").replace("万円", "").strip())
                            except ValueError:
                                return 0.0
            except Exception:
                pass
            return 0.0

        items = list(missing_prize_dams.items())
        completed = 0
        total = len(items)

        def _save_prizes():
            with open(dam_prizes_path, "w", encoding="utf-8") as f:
                json.dump(dam_prizes, f, ensure_ascii=False, indent=2)

        results = concurrent_fetch(
            items=[did for did, _ in items],
            fetch_fn=fetch_dam_prize,
            label="母馬賞金",
            save_interval=100,
            save_fn=_save_prizes,
        )

        for did in results:
            name = missing_prize_dams[did]
            prize = results[did]
            if prize is not None:
                dam_prizes[name] = prize

        _save_prizes()
        print(f"  賞金データ更新完了: {len(dam_prizes)}頭")

    # 産駒リスト取得
    if missing_foal_dams:
        print(f"\n  産駒リスト取得中...")

        def _save_foals():
            with open(dam_foals_path, "w", encoding="utf-8") as f:
                json.dump(dam_foals, f, ensure_ascii=False, indent=2)

        results = concurrent_fetch(
            items=missing_foal_dams,
            fetch_fn=fetch_dam_foal_list,
            label="産駒リスト",
            save_interval=100,
            save_fn=_save_foals,
        )

        for did, foals in results.items():
            if foals is not None:
                dam_foals[did] = foals

        _save_foals()
        print(f"  産駒リスト更新完了: {len(dam_foals)}頭")


def fetch_sire_prizes():
    """新規種牡馬の賞金データを取得する。"""
    print("\n" + "=" * 60)
    print("  Phase 4: 種牡馬賞金データ更新")
    print("=" * 60)

    sire_prizes_path = "data/sire_prizes.json"
    if os.path.exists(sire_prizes_path):
        with open(sire_prizes_path, "r", encoding="utf-8") as f:
            sire_prizes = json.load(f)
    else:
        sire_prizes = {}

    # 全年度のCSVから種牡馬名を収集
    all_sires = {}  # sire_id → sire_name
    for year in TARGET_YEARS:
        csv_path = f"data/horses_{year}.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sid = str(row.get("sire_id", ""))
            sname = str(row.get("sire", ""))
            if sid and sname and sid != "nan":
                all_sires[sid] = sname

    missing = {sid: name for sid, name in all_sires.items()
               if name not in sire_prizes}
    print(f"  全種牡馬: {len(all_sires)}頭, 未取得: {len(missing)}頭")

    if missing:
        def fetch_prize(sire_id):
            _rate_limited_sleep()
            url = f"https://db.netkeiba.com/horse/{sire_id}/"
            try:
                soup = _get_soup(url)
                for table in soup.find_all("table"):
                    for row in table.find_all("tr"):
                        th = row.find("th")
                        td = row.find("td")
                        if th and td and "本賞金" in th.text:
                            text = td.text.strip()
                            try:
                                return float(text.replace(",", "").replace("万円", "").strip())
                            except ValueError:
                                return 0.0
            except Exception:
                pass
            return 0.0

        def _save():
            with open(sire_prizes_path, "w", encoding="utf-8") as f:
                json.dump(sire_prizes, f, ensure_ascii=False, indent=2)

        results = concurrent_fetch(
            items=list(missing.keys()),
            fetch_fn=fetch_prize,
            label="種牡馬賞金",
            save_interval=50,
            save_fn=_save,
        )

        for sid, prize in results.items():
            name = missing[sid]
            if prize is not None:
                sire_prizes[name] = prize

        _save()
        print(f"  種牡馬賞金データ更新完了: {len(sire_prizes)}頭")


if __name__ == "__main__":
    print("=" * 60)
    print("  2015-2022 全データ取得")
    print("=" * 60)

    fetch_horse_lists()
    fetch_leading_data()
    fetch_dam_data()
    fetch_sire_prizes()

    print("\n" + "=" * 60)
    print("  全フェーズ完了")
    print("=" * 60)
