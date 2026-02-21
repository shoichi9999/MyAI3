# POG予測システム

競馬のPOG（ペーパーオーナーゲーム）で有望な2歳馬をランク付けするシステム。
デビュー前に入手可能な情報のみを使い、ヒューリスティックスコアでTOP10を予測する。

## スコアリング

血統スコア（父EI・母馬賞金・母父EI）をベースに、ボーナス/ペナルティを加減算する。
重みはLOO-CVグリッドサーチで最適化済み。

| 要素 | 配点 | 説明 |
|------|------|------|
| 父EI | 22.5% | 種牡馬の産駒アーニングインデックス |
| 母馬獲得賞金 | 7.5% | 母馬の現役時代の獲得賞金（対数正規化） |
| 母父EI | 25% | 母父馬の産駒EI |
| 早生まれ | +5pt | 1〜4月生まれ |
| 両親若齢 | +8pt | 父母とも13歳以下 |
| 母-母父年齢差 | +5pt | 年齢差15歳以下 |
| セリ価格 | +0〜5pt | 高額馬ほど加算 |
| 産駒番号 | -5〜+3pt | 初仔は不利、2〜4番仔が好成績 |
| 馬主 | ×0.08 | 実績スコア（50基準） |
| 生産牧場 | ×0.15 | 実績スコア（50基準） |
| 調教師 | ×0.05 | 実績スコア（50基準） |

EIデータは年度別リーディングを参照し、バックテスト時の未来データリークを防止する
（生年Yの馬 → Y+1年のリーディングを使用）。

## ファイル構成

```
run.py                          エントリーポイント
backtest.py                     バックテスト（精度検証）
grid_search.py                  LOO-CVグリッドサーチ（重み最適化）
fetch_dam_prizes.py             母馬獲得賞金の一括取得

scripts/
  fetch_leading.py              種牡馬/BMSリーディング取得（年度別）
  fetch_all_features.py         生年月日・セリ価格・産駒番号の取得

src/
  scraper.py    netkeiba.comスクレイピング（馬一覧 + 親ID/生年）
  features.py   特徴量生成（年度別EI・母馬賞金・調教師・牧場等）
  model.py      ヒューリスティックスコア算出
  predictor.py  収集→予測パイプライン

data/
  horses_YYYY.csv               馬一覧（親ID・生年含む）
  sire_leading_YYYY.json        種牡馬リーディング（年度別）
  bms_leading_YYYY.json         母父馬リーディング（年度別）
  dam_prizes.json               母馬獲得賞金
  birth_dates_YYYY.json         生年月日キャッシュ
  extra_features_YYYY.json      セリ価格・産駒番号キャッシュ
```

## 使い方

```bash
pip install -r requirements.txt

# 全自動（データ収集→予測）
python run.py --year 2024

# データ収集のみ
python run.py --mode collect --year 2024

# 予測のみ（収集済みデータ使用）
python run.py --mode predict --year 2024 --top-n 10

# リーディングデータの取得（年度別）
python scripts/fetch_leading.py 2024
python scripts/fetch_leading.py --years 2016 2017 2018 2019 2020 2021 2022

# 追加特徴量の取得（生年月日・セリ価格・産駒番号）
python scripts/fetch_all_features.py 2024

# 母馬獲得賞金の取得
python fetch_dam_prizes.py --year 2024

# バックテスト
python backtest.py 2021

# グリッドサーチ（重み最適化）
python grid_search.py
python grid_search.py --years 2018 2019 2020 2021
```
