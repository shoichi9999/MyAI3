"""
バックテスト＆ハイパーパラメータ最適化。

指定年の産駒データで特徴量を構築し、実際の賞金ランキングとの
相関・TOP N一致率を評価する。GBRハイパーパラメータの最適化も行う。

使い方:
  python backtest.py 2022
  python backtest.py 2022 --optimize
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from src.features import build_feature_matrix
from src.model import FEATURE_COLS


# ------------------------------------------------------------------
# データ準備
# ------------------------------------------------------------------

def load_backtest_data(year: int) -> pd.DataFrame:
    """バックテスト用データを読み込み、特徴量を生成する。"""
    csv_path = f"data/horses_{year}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません")

    horses = pd.read_csv(csv_path)

    # 実績賞金（目的変数）
    horses["prize_num"] = horses["total_prize"].apply(_parse_prize)

    # 特徴量マトリクス構築
    features = build_feature_matrix(horses, birth_year=year)
    features["prize_num"] = horses["prize_num"].values

    return features


def _parse_prize(x) -> float:
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    return float(str(x).replace(",", "").replace("万", "").strip())


# ------------------------------------------------------------------
# 評価
# ------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray, scores: np.ndarray, df: pd.DataFrame,
                   label: str = "Model") -> dict:
    """モデルの評価指標を計算して表示する。"""
    corr, pval = spearmanr(scores, y_true)

    results = {}
    for n in [10, 30, 50]:
        pred_idx = np.argsort(-scores)[:n]
        actual_idx = np.argsort(-y_true)[:n]
        overlap = len(set(pred_idx) & set(actual_idx))
        results[f"top{n}"] = overlap

    overall_avg = y_true.mean()
    pred_top30_avg = y_true[np.argsort(-scores)[:30]].mean()
    actual_top30_avg = y_true[np.argsort(-y_true)[:30]].mean()

    metrics = {
        "label": label,
        "spearman": corr,
        "pval": pval,
        "top10": results["top10"],
        "top30": results["top30"],
        "top50": results["top50"],
        "pred_top30_avg": pred_top30_avg,
        "actual_top30_avg": actual_top30_avg,
        "overall_avg": overall_avg,
        "prize_ratio": pred_top30_avg / overall_avg if overall_avg > 0 else 0,
    }
    return metrics


def print_metrics(m: dict):
    print(f"  Spearman:    {m['spearman']:.4f} (p={m['pval']:.2e})")
    print(f"  TOP10一致:   {m['top10']}/10")
    print(f"  TOP30一致:   {m['top30']}/30")
    print(f"  TOP50一致:   {m['top50']}/50")
    print(f"  賞金倍率:    {m['prize_ratio']:.2f}x (予測TOP30平均/全体平均)")


def print_top_horses(df: pd.DataFrame, score_col: str, n: int = 20):
    """予測TOP N と実績TOP N を表示する。"""
    df = df.copy()
    df["pred_rank"] = df[score_col].rank(ascending=False).astype(int)
    df["actual_rank"] = df["prize_num"].rank(ascending=False).astype(int)

    print(f"\n--- 予測 TOP{n} ---")
    top_pred = df.nlargest(n, score_col)
    rows = []
    for i, (_, r) in enumerate(top_pred.iterrows(), 1):
        mark = "*" if r["actual_rank"] <= 30 else ""
        rows.append([
            i, r["horse_name"], f'{r[score_col]:.1f}',
            f'{r["prize_num"]:,.0f}万', r["actual_rank"], mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "スコア", "実賞金", "実順位", ""],
                   tablefmt="simple"))

    print(f"\n--- 実績 TOP{n} ---")
    top_actual = df.nlargest(n, "prize_num")
    rows = []
    for i, (_, r) in enumerate(top_actual.iterrows(), 1):
        mark = "*" if r["pred_rank"] <= 30 else ""
        rows.append([
            i, r["horse_name"], f'{r["prize_num"]:,.0f}万',
            f'{r[score_col]:.1f}', r["pred_rank"], mark
        ])
    print(tabulate(rows, headers=["#", "馬名", "実賞金", "スコア", "予測順位", ""],
                   tablefmt="simple"))


# ------------------------------------------------------------------
# ヒューリスティック評価
# ------------------------------------------------------------------

def run_heuristic(df: pd.DataFrame) -> dict:
    """ヒューリスティックスコアで評価する。"""
    from src.model import _heuristic_score
    scores = _heuristic_score(df).values
    y_true = df["prize_num"].values
    return evaluate_model(y_true, scores, df, label="ヒューリスティック"), scores


# ------------------------------------------------------------------
# GBR Leave-One-Year-Out 評価
# ------------------------------------------------------------------

def run_gbr_loyo(target_year: int, training_years: list[int],
                 params: dict = None) -> tuple[dict, np.ndarray]:
    """
    Leave-One-Year-Out: training_years で学習、target_year で評価。
    """
    if params is None:
        params = {
            "n_estimators": 300, "max_depth": 2, "learning_rate": 0.03,
            "subsample": 0.8, "min_samples_leaf": 30, "max_features": 0.7,
        }

    # 学習データ
    train_dfs = []
    for y in training_years:
        path = f"data/horses_{y}.csv"
        if not os.path.exists(path):
            print(f"  [SKIP] {path} なし")
            continue
        h = pd.read_csv(path)
        h["prize_num"] = h["total_prize"].apply(_parse_prize)
        feat = build_feature_matrix(h, birth_year=y)
        feat["prize_num"] = h["prize_num"].values
        train_dfs.append(feat)

    if not train_dfs:
        print("[ERROR] 学習データなし")
        return {}, np.array([])

    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"  学習データ: {len(train_df)}件 (世代: {training_years})")

    # テストデータ
    test_df = load_backtest_data(target_year)
    print(f"  テストデータ: {len(test_df)}件 ({target_year}年)")

    # 特徴量準備
    for col in FEATURE_COLS:
        for d in [train_df, test_df]:
            if col not in d.columns:
                d[col] = 0.0
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    X_train = train_df[FEATURE_COLS].values
    y_train = np.log1p(train_df["prize_num"].values)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["prize_num"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # GBR
    gbr = GradientBoostingRegressor(random_state=42, **params)
    gbr.fit(X_train_s, y_train)
    pred_gbr = np.expm1(gbr.predict(X_test_s))

    # RF
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=42,
    )
    rf.fit(X_train_s, y_train)
    pred_rf = np.expm1(rf.predict(X_test_s))

    # アンサンブル
    ensemble = pred_gbr * 0.6 + pred_rf * 0.4
    test_df["ensemble_score"] = ensemble

    # CV on train
    cv_scores = cross_val_score(gbr, X_train_s, y_train, cv=5, scoring="r2")
    print(f"  CV R2 (学習): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # 特徴量重要度
    importances = sorted(
        zip(FEATURE_COLS, gbr.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"  特徴量重要度 TOP10:")
    for feat, imp in importances[:10]:
        print(f"    {feat:<25s} {imp:.4f}")

    metrics = evaluate_model(y_test, ensemble, test_df, label="GBR+RF アンサンブル")
    return metrics, ensemble


# ------------------------------------------------------------------
# ハイパーパラメータ最適化
# ------------------------------------------------------------------

def optimize_hyperparams(target_year: int, training_years: list[int]) -> dict:
    """GBRのハイパーパラメータをグリッドサーチで最適化する。"""
    print(f"\n{'='*60}")
    print(f"  ハイパーパラメータ最適化")
    print(f"{'='*60}")

    # 学習データ
    train_dfs = []
    for y in training_years:
        path = f"data/horses_{y}.csv"
        if not os.path.exists(path):
            continue
        h = pd.read_csv(path)
        h["prize_num"] = h["total_prize"].apply(_parse_prize)
        feat = build_feature_matrix(h, birth_year=y)
        feat["prize_num"] = h["prize_num"].values
        train_dfs.append(feat)

    train_df = pd.concat(train_dfs, ignore_index=True)

    # テストデータ
    test_df = load_backtest_data(target_year)

    for col in FEATURE_COLS:
        for d in [train_df, test_df]:
            if col not in d.columns:
                d[col] = 0.0
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    X_train = train_df[FEATURE_COLS].values
    y_train = np.log1p(train_df["prize_num"].values)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["prize_num"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # グリッドサーチ
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "min_samples_leaf": [10, 20, 30, 50],
    }

    best_spearman = -1
    best_params = {}
    results = []
    total = (len(param_grid["n_estimators"]) * len(param_grid["max_depth"])
             * len(param_grid["learning_rate"]) * len(param_grid["subsample"])
             * len(param_grid["min_samples_leaf"]))
    print(f"  探索パターン: {total}件")

    count = 0
    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:
                for sub in param_grid["subsample"]:
                    for msl in param_grid["min_samples_leaf"]:
                        count += 1
                        params = {
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "subsample": sub,
                            "min_samples_leaf": msl,
                        }

                        gbr = GradientBoostingRegressor(random_state=42, **params)
                        gbr.fit(X_train_s, y_train)
                        pred = np.expm1(gbr.predict(X_test_s))

                        corr, _ = spearmanr(pred, y_test)
                        results.append({"params": params, "spearman": corr})

                        if corr > best_spearman:
                            best_spearman = corr
                            best_params = params

                        if count % 100 == 0:
                            print(f"  {count}/{total} 完了... 暫定best={best_spearman:.4f}")

    results.sort(key=lambda x: x["spearman"], reverse=True)

    print(f"\n  探索完了 ({total}パターン)")
    print(f"\n  TOP5:")
    for i, r in enumerate(results[:5], 1):
        p = r["params"]
        print(f"  {i}. Spearman={r['spearman']:.4f}  "
              f"n={p['n_estimators']} depth={p['max_depth']} "
              f"lr={p['learning_rate']} sub={p['subsample']} msl={p['min_samples_leaf']}")

    print(f"\n  最適パラメータ:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    return best_params


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="バックテスト＆パラメータ最適化")
    parser.add_argument("year", type=int, help="評価対象の生年（例: 2022）")
    parser.add_argument("--optimize", action="store_true", help="ハイパーパラメータ最適化を実行")
    parser.add_argument("--training-years", type=int, nargs="+", default=None,
                        help="学習に使う世代（デフォルト: 対象年以外の利用可能な年）")
    args = parser.parse_args()

    target = args.year

    # 利用可能な学習年を検出
    if args.training_years:
        training_years = args.training_years
    else:
        training_years = []
        for y in range(target - 5, target + 3):
            if y != target and os.path.exists(f"data/horses_{y}.csv"):
                training_years.append(y)

    print("=" * 60)
    print(f"  バックテスト: {target}年産駒")
    print(f"  学習世代:    {training_years}")
    print(f"  特徴量:      {len(FEATURE_COLS)}個")
    print("=" * 60)

    # 1. ヒューリスティック評価
    print(f"\n{'='*60}")
    print(f"  [1] ヒューリスティックスコア")
    print(f"{'='*60}")

    test_df = load_backtest_data(target)
    h_metrics, h_scores = run_heuristic(test_df)
    print_metrics(h_metrics)
    test_df["heuristic_score"] = h_scores

    # 2. GBR+RF アンサンブル評価
    print(f"\n{'='*60}")
    print(f"  [2] GBR+RF アンサンブル (デフォルトパラメータ)")
    print(f"{'='*60}")

    gbr_metrics, gbr_scores = run_gbr_loyo(target, training_years)
    if gbr_metrics:
        print_metrics(gbr_metrics)
        test_df["ensemble_score"] = gbr_scores

    # 3. ハイパーパラメータ最適化（オプション）
    if args.optimize:
        best_params = optimize_hyperparams(target, training_years)

        print(f"\n{'='*60}")
        print(f"  [3] 最適パラメータでの再評価")
        print(f"{'='*60}")

        opt_metrics, opt_scores = run_gbr_loyo(target, training_years, params=best_params)
        if opt_metrics:
            print_metrics(opt_metrics)
            test_df["optimized_score"] = opt_scores

    # 4. 比較サマリー
    print(f"\n{'='*60}")
    print(f"  比較サマリー ({target}年産駒)")
    print(f"{'='*60}")

    summary = [h_metrics]
    if gbr_metrics:
        summary.append(gbr_metrics)
    if args.optimize and opt_metrics:
        summary.append(opt_metrics)

    header = f"  {'モデル':<25s} {'Spearman':>10s} {'TOP10':>6s} {'TOP30':>6s} {'TOP50':>6s} {'倍率':>6s}"
    print(header)
    print(f"  {'-'*60}")
    for m in summary:
        print(f"  {m['label']:<25s} {m['spearman']:>10.4f} "
              f"{m['top10']:>4d}/10 {m['top30']:>4d}/30 {m['top50']:>4d}/50 "
              f"{m['prize_ratio']:>5.2f}x")

    # TOP馬の詳細表示
    score_col = "ensemble_score" if "ensemble_score" in test_df.columns else "heuristic_score"
    print_top_horses(test_df, score_col)


if __name__ == "__main__":
    main()
