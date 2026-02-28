"""
LightGBM ベースの POG 予測モデル。

デビュー前特徴量から TOP10 入りを予測する 2 値分類モデルを
Leave-One-Year-Out CV で学習・評価する。
ヒューリスティックスコアをスタッキング特徴量として利用可能。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


# 学習に使う特徴量カラム
FEATURE_COLS = [
    "sire_ei",
    "bms_ei",
    "sire_2yo_ei",
    "dam_prize",
    "sire_prize",
    "trainer_score",
    "owner_score",
    "breeder_score",
    "birth_month",
    "sire_age",
    "dam_age",
    "dam_sire_age",
    "early_born",
    "sire_young",
    "dam_young",
    "both_parents_young",
    "sire_first_crop",
    "dam_bms_gap_small",
    "sale_price_log",
    "foal_number",
    "dam_breeding_age",
    "total_dam_foals",
    "sire_dam_interaction",
    "trainer_breeder_combo",
    "sibling_classic",
    "sire_classic_count",
]

# スタッキング時に追加する特徴量
STACKING_COLS = FEATURE_COLS + ["h_score"]


def _make_label(prize: pd.Series, top_n: int = 10) -> np.ndarray:
    """賞金上位 top_n 頭を正例(1)とする 2 値ラベル。"""
    threshold_idx = np.argsort(-prize.values)[:top_n]
    label = np.zeros(len(prize), dtype=int)
    label[threshold_idx] = 1
    return label


def train_predict_loyo(
    all_data: dict[int, pd.DataFrame],
    target_year: int,
    top_n: int = 10,
    lgb_params: dict | None = None,
) -> np.ndarray:
    """Leave-One-Year-Out で target_year の予測確率を返す。

    Parameters
    ----------
    all_data : dict[int, DataFrame]
        {year: feature_matrix_with_prize_num} 年度→特徴量DF
        h_score カラムがあればスタッキング特徴量として利用
    target_year : int
        予測対象年度（この年をテストに使う）
    top_n : int
        正例とする賞金上位 N 頭（デフォルト: 10）
    lgb_params : dict, optional
        LightGBM のハイパーパラメータ

    Returns
    -------
    np.ndarray
        target_year の各馬の予測確率（高いほど TOP10 入り可能性大）
    """
    if lgb_params is None:
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 30,
            "scale_pos_weight": 1.0,
            "verbose": -1,
            "seed": 42,
        }

    # スタッキング: h_scoreがあればスタッキング特徴量を使用
    use_stacking = "h_score" in all_data[target_year].columns
    feat_cols = STACKING_COLS if use_stacking else FEATURE_COLS

    # 訓練データ: target_year 以外の全年度
    train_frames = []
    for y, df in all_data.items():
        if y == target_year:
            continue
        X = df[feat_cols].copy()
        label = _make_label(df["prize_num"], top_n)
        X["__label__"] = label
        train_frames.append(X)

    train_all = pd.concat(train_frames, ignore_index=True)
    X_train = train_all[feat_cols]
    y_train = train_all["__label__"]

    # 不均衡対応: scale_pos_weight を正例/負例比率に
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    lgb_params["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0

    # テストデータ
    test_df = all_data[target_year]
    X_test = test_df[feat_cols]

    # LightGBM 学習
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=500,
    )

    # 予測確率
    proba = model.predict(X_test, num_iteration=model.best_iteration)
    return proba


def ensemble_score(
    heuristic: np.ndarray,
    ml_proba: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """ヒューリスティックと ML 予測のアンサンブル。

    両方を [0, 1] にスケーリングしてから加重平均する。

    Parameters
    ----------
    heuristic : array-like
        ヒューリスティックスコア
    ml_proba : array-like
        LightGBM の予測確率
    alpha : float
        ヒューリスティックの重み (0=ML only, 1=heuristic only)

    Returns
    -------
    np.ndarray
        アンサンブルスコア
    """
    h = np.asarray(heuristic, dtype=float)
    m = np.asarray(ml_proba, dtype=float)

    # min-max スケーリング
    h_min, h_max = h.min(), h.max()
    if h_max > h_min:
        h_scaled = (h - h_min) / (h_max - h_min)
    else:
        h_scaled = np.zeros_like(h)

    m_min, m_max = m.min(), m.max()
    if m_max > m_min:
        m_scaled = (m - m_min) / (m_max - m_min)
    else:
        m_scaled = np.zeros_like(m)

    return alpha * h_scaled + (1 - alpha) * m_scaled
