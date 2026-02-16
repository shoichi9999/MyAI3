"""
POG賞金予測モデル。

過去の世代データを学習データとして使い、
当年の2歳馬がダービーまでに獲得する賞金を予測する。

アプローチ:
1. 過去5世代（2019-2023年生まれ）のデータで学習
2. GradientBoostingRegressorで賞金を予測
3. 当年（2024年生まれ）の馬にスコアを付与
4. デビュー前の馬も血統・調教師・セリ価格から予測可能
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# 予測に使用する特徴量カラム
FEATURE_COLS = [
    "sex",
    "sire_score",
    "dam_sire_score",
    "trainer_score",
    "sale_price",
    "num_races",
    "num_wins",
    "win_rate",
    "top3_rate",
    "total_earned",
    "avg_finish",
    "best_finish",
    "avg_odds",
    "max_distance",
    "latest_weight",
    "weight_trend",
    "graded_race_wins",
    "speed_rating",
]

# デビュー前（戦績なし）の馬にも使える特徴量
PRE_DEBUT_FEATURE_COLS = [
    "sex",
    "sire_score",
    "dam_sire_score",
    "trainer_score",
    "sale_price",
]

MODEL_PATH = "models/pog_predictor.pkl"
SCALER_PATH = "models/scaler.pkl"


class POGPredictor:
    """POG賞金予測モデル。"""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        self.backup_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_cols = FEATURE_COLS

    def train(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "total_earned",
    ) -> dict:
        """
        過去データで予測モデルを学習する。

        Parameters
        ----------
        feature_df : pd.DataFrame
            特徴量マトリクス（total_earnedを含む）
        target_col : str
            予測対象のカラム名

        Returns
        -------
        dict
            学習結果の統計情報
        """
        df = feature_df.copy()

        # 欠損値処理
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        X = df[self.feature_cols].values
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))

        # スケーリング
        X_scaled = self.scaler.fit_transform(X)

        # 学習
        self.model.fit(X_scaled, y)
        self.backup_model.fit(X_scaled, y)
        self.is_fitted = True

        # クロスバリデーション
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=min(5, len(df)), scoring="r2"
        )

        # 特徴量重要度
        importances = dict(
            zip(self.feature_cols, self.model.feature_importances_)
        )
        importances = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )

        stats = {
            "n_samples": len(df),
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "feature_importances": importances,
        }

        print(f"学習完了: {stats['n_samples']}件")
        print(f"CV R2: {stats['cv_r2_mean']:.3f} (+/- {stats['cv_r2_std']:.3f})")
        print("特徴量重要度 TOP5:")
        for i, (feat, imp) in enumerate(importances.items()):
            if i >= 5:
                break
            print(f"  {feat}: {imp:.4f}")

        return stats

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        予測スコアを算出する。

        Parameters
        ----------
        feature_df : pd.DataFrame
            特徴量マトリクス

        Returns
        -------
        pd.DataFrame
            予測スコア付きのDataFrame
        """
        df = feature_df.copy()

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        X = df[self.feature_cols].values

        if self.is_fitted:
            X_scaled = self.scaler.transform(X)
            df["predicted_prize"] = self.model.predict(X_scaled)
            df["predicted_prize_rf"] = self.backup_model.predict(X_scaled)
            # アンサンブル（加重平均）
            df["ensemble_score"] = (
                df["predicted_prize"] * 0.6 + df["predicted_prize_rf"] * 0.4
            )
        else:
            # モデル未学習の場合はヒューリスティックスコアを使用
            df["ensemble_score"] = _heuristic_score(df)

        return df

    def save(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """モデルを保存する。"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "backup": self.backup_model}, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"モデルを保存しました: {model_path}")

    def load(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """モデルを読み込む。"""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.backup_model = data["backup"]
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"モデルを読み込みました: {model_path}")


def _heuristic_score(df: pd.DataFrame) -> pd.Series:
    """
    モデル未学習時のヒューリスティックスコア。
    POGで重要な要素を重み付けして算出する。

    重み配分:
    - 血統（父）: 30%
    - 戦績（既走馬の場合）: 25%
    - 調教師: 20%
    - セリ価格: 15%
    - 血統（母父）: 10%
    """
    score = pd.Series(0.0, index=df.index)

    # 血統スコア（父）
    if "sire_score" in df.columns:
        score += df["sire_score"].fillna(50) * 0.30

    # 血統スコア（母父）
    if "dam_sire_score" in df.columns:
        score += df["dam_sire_score"].fillna(50) * 0.10

    # 調教師スコア
    if "trainer_score" in df.columns:
        score += df["trainer_score"].fillna(50) * 0.20

    # セリ価格（正規化して加算）
    if "sale_price" in df.columns:
        prices = df["sale_price"].fillna(0)
        max_price = prices.max()
        if max_price > 0:
            score += (prices / max_price) * 100 * 0.15
        else:
            score += 50 * 0.15

    # 戦績スコア
    if "total_earned" in df.columns:
        earned = df["total_earned"].fillna(0)
        max_earned = earned.max()
        if max_earned > 0:
            score += (earned / max_earned) * 100 * 0.25
        else:
            score += 50 * 0.25

    return score
