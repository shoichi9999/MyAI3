# POG予測システム

競馬のPOG（ペーパーオーナーゲーム）で、ダービーまでの期間に獲得賞金が高くなる馬をTOP10予測するシステム。

## システム構成

```
src/
  scraper.py    - netkeiba.comから馬データをスクレイピング
  features.py   - 特徴量エンジニアリング（血統EI・母馬賞金・調教師・牧場等）
  model.py      - GradientBoosting + RandomForest アンサンブル予測モデル
  predictor.py  - メイン実行ロジック（収集→学習→予測パイプライン）
scripts/
  fetch_birth_dates.py   - 生年月日データの一括取得
  fetch_progeny_data.py  - 血統・親年齢データの一括取得
  fetch_extra_features.py - セリ価格・産駒番号の一括取得
run.py              - エントリーポイント
backtest.py         - バックテスト＆ハイパーパラメータ最適化
fetch_dam_prizes.py - 母馬獲得賞金の一括取得
```

## 予測で使用する特徴量（19個）

デビュー前に入手可能な情報のみで予測する。

| カテゴリ | 特徴量 | 説明 |
|---------|--------|------|
| 血統 | sire_ei | 父馬の産駒EI（アーニングインデックス） |
| 血統 | bms_ei | 母父馬の産駒EI |
| 血統 | dam_prize_log | 母馬の獲得賞金（対数変換） |
| 血統 | sire_dam_interaction | 父EI × 母賞金の交互作用 |
| 人的要素 | trainer_score | 調教師の2歳戦・クラシック実績スコア |
| 人的要素 | owner_score | 馬主の重賞実績スコア |
| 人的要素 | breeder_score | 生産牧場の実績スコア |
| 親年齢 | sire_age | 父馬の産駒時年齢 |
| 親年齢 | dam_age | 母馬の産駒時年齢 |
| バイナリ | early_born | 1-4月生まれ=1 |
| バイナリ | sire_young | 父13歳以下=1 |
| バイナリ | dam_young | 母13歳以下=1 |
| バイナリ | both_parents_young | 両親とも13歳以下=1 |
| バイナリ | sire_first_crop | 父の初期産駒（sire_age<=7）=1 |
| バイナリ | dam_bms_gap_small | 母と母父の年齢差15以下=1 |
| 市場 | sale_price_log | セリ取引価格（対数変換） |
| 市場 | foal_number | 何番仔か |
| 基本 | sex | 性別（牡=1/牝=0/セン=0.5） |
| 基本 | birth_month | 生まれ月（1-12） |

### 特徴量設計の方針

- 祖父母・曽祖父母の年齢統計量（gp_age_std, ggp_age_std等）は除外。欠損時のデフォルト値が「データ有無」のプロキシとなり、GBR特徴量重要度の99%以上を独占する問題があったため
- 牧場品質は `breeder_score` で明示的に捕捉
- `max_features=0.7` で各分割時の特徴量をサブサンプリングし、単一特徴量の支配を緩和

## データソース

- `data/sire_leading_2024.json` - 種牡馬リーディング（産駒賞金・EI）
- `data/bms_leading_2024.json` - 母父馬リーディング（産駒賞金・EI）
- `data/dam_prizes.json` - 母馬の獲得賞金キャッシュ
- `data/horses_YYYY.csv` - 各世代の馬データ
- `data/birth_dates_YYYY.json` - 生年月日キャッシュ
- `data/parent_ages_YYYY.json` - 親・祖父母・曽祖父母の生年キャッシュ
- `data/extra_features_YYYY.json` - セリ価格・産駒番号キャッシュ

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# 全自動実行（データ収集→学習→予測）
python run.py --year 2024

# テスト実行（少数の馬で動作確認）
python run.py --year 2024 --max-horses 20

# データ収集のみ
python run.py --mode collect --year 2024

# 予測のみ（収集済みデータ使用）
python run.py --mode predict --year 2024 --top-n 10

# バックテスト
python backtest.py 2022

# バックテスト＋ハイパーパラメータ最適化
python backtest.py 2022 --optimize
```

## 予測アプローチ

1. **データ収集**: netkeiba.comから馬のプロフィール・血統・親年齢情報を取得
2. **特徴量生成**: 産駒EI、母馬獲得賞金、調教師/馬主/牧場スコア、親年齢バイナリ等を算出
3. **モデル学習**: 過去世代のデータでGBR + RFアンサンブルモデルを学習（目的変数は対数変換）
4. **予測**: 当年世代にスコアを付与し、上位10頭を選出

### モデル構成

- **GradientBoostingRegressor** (60%): n_estimators=300, max_depth=2, lr=0.03, max_features=0.7
- **RandomForestRegressor** (40%): n_estimators=200, max_depth=4
- **前処理**: StandardScaler正規化、目的変数はlog1p変換

## バックテスト結果

### 2022年産駒

学習データ: 2023年・2024年産駒（計4,000頭）、テスト: 2022年産駒（2,000頭）

| 指標 | ヒューリスティック | GBR+RFアンサンブル |
|------|-------------------|-------------------|
| Spearman順位相関 | 0.195 | 0.187 |
| TOP10一致 | 0/10 | 0/10 |
| TOP30一致 | 0/30 | 3/30 |
| TOP50一致 | 4/50 | 8/50 |
| 予測TOP30の賞金倍率 | 1.18x | **3.69x** |

### 主な的中馬（2022年産駒）

| 馬名 | 実績順位 | 予測順位 | 実賞金 |
|------|---------|---------|--------|
| ミュージアムマイル | 1位 | **12位** | 96,180万円 |
| クロワデュノール | 3位 | **18位** | 60,749万円 |
| ゲルチュタール | 12位 | **22位** | 15,938万円 |

### 特徴量重要度（GBR）

| 特徴量 | 重要度 |
|--------|--------|
| dam_age | 60.2% |
| sire_age | 31.0% |
| both_parents_young | 6.8% |
| birth_month | 1.6% |
| sire_ei | 0.2% |
| その他 | 0.2% |
