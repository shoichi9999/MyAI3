"""
POG賞金予測モデル。

過去の世代データを学習データとして使い、
当年の2歳馬がダービーまでに獲得する賞金を予測する。

アプローチ:
1. 過去5世代（2019-2023年生まれ）のデータで学習
2. GradientBoostingRegressorで賞金を予測
3. 当年（2024年生まれ）の馬にスコアを付与
4. デビュー前の馬も血統情報（父EI・母父EI・母馬賞金）＋生まれ月＋親年齢＋祖父母年齢から予測可能
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# 予測に使用する特徴量カラム（血統＋生まれ月＋親年齢＋祖父母年齢）
FEATURE_COLS = [
    "sex",
    "sire_ei",
    "bms_ei",
    "dam_prize",
    "birth_month",
    "sire_age",
    "dam_age",
    "sire_sire_age",
    "sire_dam_age",
    "dam_sire_age",
    "dam_dam_age",
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

    血統スコア（父EI 40% + 母馬賞金 40% + 母父EI 20%）に
    生まれ月ボーナスと親年齢ボーナスを加算する。
    """
    from src.features import WEIGHT_SIRE_EI, WEIGHT_DAM_PRIZE, WEIGHT_BMS_EI

    score = pd.Series(0.0, index=df.index)

    # 父EI（正規化）
    if "sire_ei" in df.columns:
        ei = df["sire_ei"].fillna(0)
        max_ei = ei.max()
        if max_ei > 0:
            score += (ei / max_ei) * 100 * WEIGHT_SIRE_EI

    # 母父EI（正規化）
    if "bms_ei" in df.columns:
        ei = df["bms_ei"].fillna(0)
        max_ei = ei.max()
        if max_ei > 0:
            score += (ei / max_ei) * 100 * WEIGHT_BMS_EI

    # 母馬獲得賞金（対数正規化）
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        max_dp = dp.max()
        if max_dp > 0:
            score += (dp / max_dp) * 100 * WEIGHT_DAM_PRIZE

    # 生まれ月ボーナス（早生まれほど有利）
    if "birth_month" in df.columns:
        month_bonus = df["birth_month"].map({
            1: 8, 2: 5, 3: 2, 4: 0, 5: -4, 6: -6,
        }).fillna(-2)
        score += month_bonus

    # 父年齢ボーナス（若い父ほど有利）
    if "sire_age" in df.columns:
        sa = df["sire_age"].fillna(11)
        score += np.where(sa <= 10, 4, np.where(sa <= 13, 2, np.where(sa <= 16, 0, -3)))

    # 母年齢ボーナス（若い母ほど有利）
    if "dam_age" in df.columns:
        da = df["dam_age"].fillna(10.5)
        score += np.where(da <= 8, 4, np.where(da <= 11, 2, np.where(da <= 14, 0, -3)))

    # 祖父母年齢ボーナス（適齢の祖父母が有利）
    for col, default_val in [
        ("sire_sire_age", 22.0),
        ("sire_dam_age", 21.0),
        ("dam_sire_age", 22.0),
        ("dam_dam_age", 21.0),
    ]:
        if col in df.columns:
            age = df[col].fillna(default_val)
            score += np.where(age <= 20, 2, np.where(age <= 25, 1, np.where(age <= 30, 0, -1)))

    return score
