"""
POG賞金予測モデル。

過去の世代データを学習データとして使い、
当年の2歳馬がダービーまでに獲得する賞金を予測する。

アプローチ:
1. 過去5世代（2019-2023年生まれ）のデータで学習
2. GradientBoostingRegressorで賞金を予測
3. 当年（2024年生まれ）の馬にスコアを付与
4. デビュー前の馬も血統情報（父EI・母父EI・母馬賞金）＋生まれ月＋親年齢＋世代別年齢統計量から予測可能
5. 目的変数は対数変換、祖父母/曾祖父母年齢は世代別統計量に集約（最適化済み）
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# 予測に使用する特徴量カラム（最適化済み: 14個）
# Ablation分析で除外: ggp_age_mean(gp_age_meanと相関0.91), breeder_score(owner_scoreと相関0.52)
FEATURE_COLS = [
    "sex",
    "sire_ei",
    "bms_ei",
    "dam_prize_log",
    "birth_month",
    "trainer_score",
    "owner_score",
    # 親の産駒時年齢
    "sire_age",
    "dam_age",
    # ドメイン知識ベース（閾値バイナリ）
    "early_born",          # 1-4月生まれ=1
    "sire_young",          # 父13歳以下=1
    "dam_young",           # 母13歳以下=1
    "both_parents_young",  # 両親とも13歳以下=1
    "sire_first_crop",     # 父の初期産駒（sire_age<=7）=1
    "dam_bms_gap_small",   # 母と母父の年齢差15以下=1
    # 追加特徴量
    "sale_price_log",      # セリ取引価格（対数）
    "foal_number",         # 何番仔か
    # 祖父母年齢の集約統計量
    "gp_age_mean",
    "gp_age_min",
    "gp_age_std",
    # 曾祖父母年齢の集約統計量（meanは祖父母と冗長なので除外）
    "ggp_age_min",
    "ggp_age_std",
]

MODEL_PATH = "models/pog_predictor.pkl"
SCALER_PATH = "models/scaler.pkl"


class POGPredictor:
    """POG賞金予測モデル。"""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=30,
            random_state=42,
        )
        self.backup_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.log_target = True  # 目的変数を対数変換
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

        # 目的変数の対数変換（極端な賞金分布を安定化）
        if self.log_target:
            y = np.log1p(y)

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
            pred_main = self.model.predict(X_scaled)
            pred_rf = self.backup_model.predict(X_scaled)

            # 対数空間で予測した場合は逆変換
            if self.log_target:
                df["predicted_prize"] = np.expm1(pred_main)
                df["predicted_prize_rf"] = np.expm1(pred_rf)
            else:
                df["predicted_prize"] = pred_main
                df["predicted_prize_rf"] = pred_rf

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
            pickle.dump({"model": self.model, "backup": self.backup_model, "log_target": self.log_target}, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"モデルを保存しました: {model_path}")

    def load(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """モデルを読み込む。"""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.backup_model = data["backup"]
            self.log_target = data.get("log_target", True)
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

    # 調教師スコアボーナス
    if "trainer_score" in df.columns:
        ts = df["trainer_score"].fillna(50)
        score += (ts - 50) * 0.2  # 50基準で加減算

    # 馬主スコアボーナス
    if "owner_score" in df.columns:
        os_val = df["owner_score"].fillna(50)
        score += (os_val - 50) * 0.2

    # 母馬獲得賞金（対数正規化）
    if "dam_prize" in df.columns:
        dp = np.log1p(df["dam_prize"].fillna(0))
        max_dp = dp.max()
        if max_dp > 0:
            score += (dp / max_dp) * 100 * WEIGHT_DAM_PRIZE

    # 早生まれボーナス（1-4月生まれ）
    if "early_born" in df.columns:
        score += df["early_born"].fillna(0) * 5

    # 両親若齢ボーナス（13歳以下）
    if "both_parents_young" in df.columns:
        score += df["both_parents_young"].fillna(0) * 5
    elif "sire_young" in df.columns and "dam_young" in df.columns:
        score += (df["sire_young"].fillna(0) + df["dam_young"].fillna(0)) * 2.5

    # 母と母父の年齢差が小さいボーナス
    if "dam_bms_gap_small" in df.columns:
        score += df["dam_bms_gap_small"].fillna(0) * 3

    # セリ価格ボーナス（高額馬ほど有利）
    if "sale_price_log" in df.columns:
        sp = df["sale_price_log"].fillna(0)
        max_sp = sp.max()
        if max_sp > 0:
            score += (sp / max_sp) * 8

    # 産駒番号（初仔は不利、2-4番仔がスイートスポット）
    if "foal_number" in df.columns:
        fn = df["foal_number"].fillna(3)
        score += np.where(fn == 1, -3, np.where(fn <= 4, 2, 0))

    # 父年齢ボーナス（若い父ほど有利）
    if "sire_age" in df.columns:
        sa = df["sire_age"].fillna(11)
        score += np.where(sa <= 10, 4, np.where(sa <= 13, 2, np.where(sa <= 16, 0, -3)))

    # 祖父母年齢ボーナス（集約: 平均年齢が若いほど有利）
    if "gp_age_mean" in df.columns:
        gp = df["gp_age_mean"].fillna(21.5)
        score += np.where(gp <= 20, 3, np.where(gp <= 23, 1.5, np.where(gp <= 27, 0, -2)))

    # 祖父母年齢の散布度ボーナス（均質な世代構成が有利）
    if "gp_age_std" in df.columns:
        gp_std = df["gp_age_std"].fillna(3.0)
        score += np.where(gp_std <= 2, 1, np.where(gp_std <= 4, 0, -1))

    # 曾祖父母年齢ボーナス（最小年齢）
    if "ggp_age_min" in df.columns:
        ggp_min = df["ggp_age_min"].fillna(25.0)
        score += np.where(ggp_min <= 22, 2, np.where(ggp_min <= 28, 1, np.where(ggp_min <= 35, 0, -1)))

    return score
