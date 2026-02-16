# POG予測システム

競馬のPOG（ペーパーオーナーゲーム）で、ダービーまでの期間に獲得賞金が高くなる馬をTOP10予測するシステム。

## システム構成

```
src/
  scraper.py    - netkeiba.comから2歳馬データをスクレイピング
  features.py   - 特徴量エンジニアリング（血統・調教師・戦績等）
  model.py      - GradientBoosting + RandomForest アンサンブル予測モデル
  predictor.py  - メイン実行ロジック（収集→学習→予測パイプライン）
run.py          - エントリーポイント
```

## 予測で使用する特徴量

| カテゴリ | 特徴量 | 説明 |
|---------|--------|------|
| 血統 | sire_score | 父馬の産駒成績スコア |
| 血統 | dam_sire_score | 母父馬のスコア |
| 調教師 | trainer_score | 調教師の2歳戦・クラシック実績 |
| 取引 | sale_price | セリ取引価格 |
| 戦績 | win_rate, top3_rate | 勝率・3着以内率 |
| 戦績 | speed_rating | タイムベースのスピード指数 |
| 馬体 | latest_weight, weight_trend | 馬体重と成長傾向 |

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
```

## 予測アプローチ

1. **データ収集**: netkeiba.comから2歳馬のプロフィール・血統・戦績を取得
2. **特徴量生成**: 血統スコア、調教師スコア、戦績指標、馬体重傾向などを算出
3. **モデル学習**: 過去5世代のデータでGBR + RFアンサンブルモデルを学習
4. **予測**: 当年世代にスコアを付与し、上位10頭を選出

デビュー前の馬は血統・調教師・セリ価格のヒューリスティックスコアで評価。
既走馬は戦績ベースの特徴量も加味して予測精度を向上させる。
