# POG予測システム

競馬のPOG（ペーパーオーナーゲーム）で、ダービーまでの期間に獲得賞金が高くなる馬をTOP10予測するシステム。

## システム構成

```
src/
  scraper.py    - netkeiba.comから馬データをスクレイピング
  features.py   - 特徴量エンジニアリング（血統EI・母馬賞金・調教師）
  model.py      - GradientBoosting + RandomForest アンサンブル予測モデル
  predictor.py  - メイン実行ロジック（収集→学習→予測パイプライン）
run.py          - エントリーポイント
backtest_2023.py - バックテスト（2023年世代）
fetch_dam_prizes.py - 母馬獲得賞金の一括取得
```

## 予測で使用する特徴量

デビュー前に入手可能な情報のみで予測する。

| カテゴリ | 特徴量 | 説明 |
|---------|--------|------|
| 血統 | sire_ei | 父馬の産駒EI（アーニングインデックス） |
| 血統 | bms_ei | 母父馬の産駒EI |
| 血統 | dam_prize | 母馬の現役時の獲得賞金 |
| 調教師 | trainer_score | 調教師の2歳戦・クラシック実績スコア |
| 基本 | sex | 性別（牡/牝/セン） |

### 重み配分（バックテスト最良のE2構成）

- 母馬獲得賞金: 30%
- 調教師: 30%
- 父EI: 25%
- 母父EI: 15%

## データソース

- `data/sire_leading_2024.json` - 種牡馬リーディング（産駒賞金・EI）
- `data/bms_leading_2024.json` - 母父馬リーディング（産駒賞金・EI）
- `data/dam_prizes.json` - 母馬の獲得賞金キャッシュ
- `data/horses_YYYY.csv` - 各世代の馬データ

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
python backtest_2023.py
```

## 予測アプローチ

1. **データ収集**: netkeiba.comから馬のプロフィール・血統情報を取得
2. **特徴量生成**: 産駒EI、母馬獲得賞金、調教師スコアを算出
3. **モデル学習**: 過去世代のデータでGBR + RFアンサンブルモデルを学習
4. **予測**: 当年世代にスコアを付与し、上位10頭を選出

## バックテスト結果

### 2022年世代（暫定・母馬カバー21%）

| 指標 | 結果 |
|------|------|
| Spearman順位相関 | 0.183 (p<10^-16) |
| TOP30一致 | 1/30 |
| 予測TOP30の賞金倍率 | 1.6x（全体平均比） |
