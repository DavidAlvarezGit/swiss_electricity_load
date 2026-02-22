import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_training_file(input_path=None):
    """Find training dataset file (.parquet preferred, then .csv)."""
    if input_path is None:
        base_dir = Path("data/processed")
    else:
        candidate = Path(input_path)
        if candidate.is_file():
            return candidate
        base_dir = candidate

    parquet_file = base_dir / "training_dataset_quarter_hourly.parquet"
    csv_file = base_dir / "training_dataset_quarter_hourly.csv"

    if parquet_file.exists():
        return parquet_file
    if csv_file.exists():
        return csv_file

    raise FileNotFoundError(
        "Could not find training_dataset_quarter_hourly.parquet or training_dataset_quarter_hourly.csv"
    )


def load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def choose_target_column(df, target_column=None):
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column not found: {target_column}")
        return target_column

    if "load" in df.columns and pd.api.types.is_numeric_dtype(df["load"]):
        return "load"

    for col in df.columns:
        if col == "timestamp":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col

    raise ValueError("No numeric target column found")


def split_time_order(df, test_fraction=0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_fraction))

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Not enough rows for train/test split")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def get_feature_columns(df, target_column):
    feature_columns = []
    for col in df.columns:
        if col in ["timestamp", target_column]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    return feature_columns


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def baseline_predict(test_df, target_column):
    lag_col = f"{target_column}_lag_1"
    if lag_col in test_df.columns:
        return test_df[lag_col].to_numpy(dtype=float)

    # Fallback: naive persistence on the target itself.
    values = test_df[target_column].shift(1).bfill()
    return values.to_numpy(dtype=float)


def fit_linear_model(train_df, feature_columns, target_column):
    x = train_df[feature_columns].to_numpy(dtype=float)
    y = train_df[target_column].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    return coefficients


def predict_linear_model(df, feature_columns, coefficients):
    x = df[feature_columns].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    return x_design @ coefficients


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def add_quantile_interval(train_true, train_pred, test_pred, alpha=0.1):
    """Create prediction interval using residual quantiles from training set."""
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")

    residuals = np.asarray(train_true) - np.asarray(train_pred)
    lower_shift = float(np.quantile(residuals, alpha))
    upper_shift = float(np.quantile(residuals, 1 - alpha))

    lower = np.asarray(test_pred) + lower_shift
    upper = np.asarray(test_pred) + upper_shift
    return lower, upper


def _import_lightgbm():
    """Import LightGBM with a clear beginner-friendly error if missing."""
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise ImportError(
            "LightGBM is not installed. Run: poetry add lightgbm\n"
            "Then run training again with --use-lightgbm."
        ) from exc
    return lgb


def train_lightgbm_point_model(train_df, test_df, feature_columns, target_column, random_state=42):
    """Train one standard LightGBM regressor and return test predictions."""
    lgb = _import_lightgbm()

    # Use numpy arrays to avoid fragile feature-name issues on raw source columns.
    x_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy(dtype=float)
    x_test = test_df[feature_columns].to_numpy(dtype=float)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return model, preds


def train_lightgbm_quantile_models(
    train_df,
    test_df,
    feature_columns,
    target_column,
    quantiles=(0.1, 0.5, 0.9),
    random_state=42,
):
    """
    Train one LightGBM model per quantile.
    Example quantiles: (0.1, 0.5, 0.9)
    """
    lgb = _import_lightgbm()

    # Use numpy arrays to avoid fragile feature-name issues on raw source columns.
    x_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy(dtype=float)
    x_test = test_df[feature_columns].to_numpy(dtype=float)

    models = {}
    preds_by_q = {}

    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        models[q] = model
        preds_by_q[q] = model.predict(x_test)

    return models, preds_by_q


def save_lightgbm_model(model, output_path):
    """
    Save a LightGBM model to disk.
    Works with sklearn wrapper (model.booster_) or booster object directly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "booster_") and model.booster_ is not None:
        model.booster_.save_model(str(output_path))
        return
    if hasattr(model, "save_model"):
        model.save_model(str(output_path))
        return

    raise ValueError("Could not save LightGBM model: unsupported model object")


def make_time_series_folds(df, n_folds=3, test_fraction=0.2, min_train_fraction=0.5):
    """
    Build rolling time-series folds (expanding train window, fixed test window).
    Returns list of (train_df, test_df).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 < min_train_fraction < 1:
        raise ValueError("min_train_fraction must be between 0 and 1")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n_rows = len(df)
    test_size = int(n_rows * test_fraction)
    min_train_size = int(n_rows * min_train_fraction)

    if test_size < 1:
        raise ValueError("Not enough rows to build test folds")
    if min_train_size < 1:
        raise ValueError("Not enough rows to build training folds")

    max_folds = (n_rows - min_train_size) // test_size
    if max_folds < 1:
        raise ValueError("Not enough rows for time-series cross-validation")

    n_folds = min(n_folds, max_folds)
    folds = []

    for fold in range(n_folds):
        test_end = n_rows - (n_folds - 1 - fold) * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train_size:
            continue

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        if len(train_df) > 0 and len(test_df) > 0:
            folds.append((train_df, test_df))

    if not folds:
        raise ValueError("Could not build valid time-series CV folds")

    return folds


def average_metric_dicts(metric_dicts):
    """Average a list of metric dictionaries."""
    keys = metric_dicts[0].keys()
    averaged = {}
    for key in keys:
        values = [m[key] for m in metric_dicts if m[key] is not None]
        averaged[key] = float(np.mean(values)) if values else None
    return averaged


def run_time_series_cv(
    df,
    target_column,
    feature_columns,
    n_folds=3,
    test_fraction=0.2,
    min_train_fraction=0.5,
    use_lightgbm=False,
):
    """
    Run rolling time-series cross-validation.
    Evaluates baseline + linear, and LightGBM if requested.
    """
    folds = make_time_series_folds(
        df,
        n_folds=n_folds,
        test_fraction=test_fraction,
        min_train_fraction=min_train_fraction,
    )

    per_fold = []
    baseline_metrics_all = []
    linear_metrics_all = []
    lightgbm_metrics_all = []

    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        y_test = test_df[target_column].to_numpy(dtype=float)

        baseline_pred = baseline_predict(test_df, target_column)
        baseline_metrics = evaluate_predictions(y_test, baseline_pred)
        baseline_metrics_all.append(baseline_metrics)

        coefficients = fit_linear_model(train_df, feature_columns, target_column)
        linear_pred = predict_linear_model(test_df, feature_columns, coefficients)
        linear_metrics = evaluate_predictions(y_test, linear_pred)
        linear_metrics_all.append(linear_metrics)

        row = {
            "fold": fold_id,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "baseline_metrics": baseline_metrics,
            "linear_metrics": linear_metrics,
            "lightgbm_metrics": None,
        }

        if use_lightgbm:
            _, lgb_pred = train_lightgbm_point_model(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            lgb_metrics = evaluate_predictions(y_test, lgb_pred)
            row["lightgbm_metrics"] = lgb_metrics
            lightgbm_metrics_all.append(lgb_metrics)

        per_fold.append(row)

    result = {
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "mean_baseline_metrics": average_metric_dicts(baseline_metrics_all),
        "mean_linear_metrics": average_metric_dicts(linear_metrics_all),
        "mean_lightgbm_metrics": average_metric_dicts(lightgbm_metrics_all) if lightgbm_metrics_all else None,
    }
    return result


def train_models(
    input_path=None,
    output_dir="data/processed",
    target_column=None,
    test_fraction=0.2,
    make_quantile_interval=False,
    alpha=0.1,
    use_lightgbm=False,
    use_lightgbm_quantiles=False,
    cv_folds=0,
    cv_test_fraction=None,
    cv_min_train_fraction=0.5,
    save_models=True,
):
    """
    Train two beginner-friendly models:
    1) baseline (lag-1)
    2) linear regression (advanced but still simple)

    Optionally:
    - build a residual-based interval for linear model
    - train LightGBM point model
    - train LightGBM quantile models (P10/P50/P90)
    """
    source_path = find_training_file(input_path)
    df = load_table(source_path)

    if "timestamp" not in df.columns:
        raise ValueError("Training dataset must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    target = choose_target_column(df, target_column=target_column)
    feature_columns = get_feature_columns(df, target)

    train_df, test_df = split_time_order(df, test_fraction=test_fraction)

    y_test = test_df[target].to_numpy(dtype=float)

    # Baseline: previous value
    baseline_pred = baseline_predict(test_df, target)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)

    # Advanced: linear regression via least squares
    coefficients = fit_linear_model(train_df, feature_columns, target)
    linear_pred = predict_linear_model(test_df, feature_columns, coefficients)
    linear_metrics = evaluate_predictions(y_test, linear_pred)

    prediction_table = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].values,
            "y_true": y_test,
            "baseline_pred": baseline_pred,
            "linear_pred": linear_pred,
        }
    )

    quantile_info = None
    if make_quantile_interval:
        train_pred = predict_linear_model(train_df, feature_columns, coefficients)
        lower, upper = add_quantile_interval(
            train_true=train_df[target].to_numpy(dtype=float),
            train_pred=train_pred,
            test_pred=linear_pred,
            alpha=alpha,
        )
        prediction_table["linear_q_lower"] = lower
        prediction_table["linear_q_upper"] = upper
        quantile_info = {"enabled": True, "alpha": alpha}

    report = {
        "input_file": str(source_path),
        "target_column": target,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "baseline_metrics": baseline_metrics,
        "linear_metrics": linear_metrics,
        "lightgbm_metrics": None,
        "quantile": quantile_info,
        "lightgbm_quantiles": None,
        "time_series_cv": None,
        "saved_models": [],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"

    if use_lightgbm:
        point_model, lgb_pred = train_lightgbm_point_model(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target,
        )
        prediction_table["lightgbm_pred"] = lgb_pred
        report["lightgbm_metrics"] = evaluate_predictions(y_test, lgb_pred)
        if save_models:
            point_path = models_dir / "lightgbm_point.txt"
            save_lightgbm_model(point_model, point_path)
            report["saved_models"].append(str(point_path))

    if use_lightgbm_quantiles:
        quantile_models, quantile_preds = train_lightgbm_quantile_models(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target,
            quantiles=(0.1, 0.5, 0.9),
        )
        prediction_table["lightgbm_q10"] = quantile_preds[0.1]
        prediction_table["lightgbm_q50"] = quantile_preds[0.5]
        prediction_table["lightgbm_q90"] = quantile_preds[0.9]
        report["lightgbm_quantiles"] = {"enabled": True, "quantiles": [0.1, 0.5, 0.9]}
        if save_models:
            for q, model in quantile_models.items():
                q_name = int(q * 100)
                q_path = models_dir / f"lightgbm_quantile_q{q_name}.txt"
                save_lightgbm_model(model, q_path)
                report["saved_models"].append(str(q_path))

    if cv_folds >= 2:
        effective_cv_test_fraction = test_fraction if cv_test_fraction is None else cv_test_fraction
        report["time_series_cv"] = run_time_series_cv(
            df=df,
            target_column=target,
            feature_columns=feature_columns,
            n_folds=cv_folds,
            test_fraction=effective_cv_test_fraction,
            min_train_fraction=cv_min_train_fraction,
            use_lightgbm=use_lightgbm,
        )

    report_path = output_dir / "model_report.json"
    pred_csv_path = output_dir / "model_predictions.csv"
    pred_parquet_path = output_dir / "model_predictions.parquet"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    prediction_table.to_csv(pred_csv_path, index=False)
    prediction_table.to_parquet(pred_parquet_path, index=False)

    return report, prediction_table


def build_parser():
    parser = argparse.ArgumentParser(description="Train baseline + linear model on training dataset")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--quantile", action="store_true", help="Add quantile interval around linear prediction")
    parser.add_argument("--alpha", type=float, default=0.1, help="Quantile alpha, e.g. 0.1 for 10%%-90%% interval")
    parser.add_argument("--use-lightgbm", action="store_true", help="Train a LightGBM point forecast model")
    parser.add_argument(
        "--use-lightgbm-quantiles",
        action="store_true",
        help="Train LightGBM quantile models (P10/P50/P90)",
    )
    parser.add_argument("--cv-folds", type=int, default=0, help="Rolling time-series CV folds (e.g. 3)")
    parser.add_argument("--cv-test-fraction", type=float, default=None, help="Test fraction per CV fold (default: same as --test-fraction)")
    parser.add_argument("--cv-min-train-fraction", type=float, default=0.5, help="Minimum training fraction for CV folds")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save trained LightGBM model files")
    return parser


def main():
    args = build_parser().parse_args()
    report, _ = train_models(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_fraction=args.test_fraction,
        make_quantile_interval=args.quantile,
        alpha=args.alpha,
        use_lightgbm=args.use_lightgbm,
        use_lightgbm_quantiles=args.use_lightgbm_quantiles,
        cv_folds=args.cv_folds,
        cv_test_fraction=args.cv_test_fraction,
        cv_min_train_fraction=args.cv_min_train_fraction,
        save_models=not args.no_save_models,
    )

    print("Training complete")
    print("Target:", report["target_column"])
    print("Baseline MAE:", report["baseline_metrics"]["mae"])
    print("Linear MAE:", report["linear_metrics"]["mae"])
    if report["lightgbm_metrics"] is not None:
        print("LightGBM MAE:", report["lightgbm_metrics"]["mae"])
    if report["time_series_cv"] is not None:
        print("CV folds:", report["time_series_cv"]["n_folds"])
        print("CV linear MAE:", report["time_series_cv"]["mean_linear_metrics"]["mae"])
        if report["time_series_cv"]["mean_lightgbm_metrics"] is not None:
            print("CV LightGBM MAE:", report["time_series_cv"]["mean_lightgbm_metrics"]["mae"])


if __name__ == "__main__":
    main()
