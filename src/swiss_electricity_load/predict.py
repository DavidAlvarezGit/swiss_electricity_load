from pathlib import Path

import pandas as pd

from swiss_electricity_load.model import find_training_file, load_table


def _format_horizon_suffix(horizon_steps, output_suffix=None):
    if output_suffix is not None:
        return output_suffix
    if horizon_steps <= 0:
        return ""
    minutes = horizon_steps * 15
    if minutes % 60 == 0:
        return f"_h{minutes // 60}"
    return f"_m{minutes}"


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise ImportError(
            "LightGBM is not installed. Run: poetry install (or poetry add lightgbm) first."
        ) from exc
    return lgb


def load_json(path):
    return Path(path).read_text(encoding="utf-8")


def parse_feature_columns(report_path):
    import json

    report_text = load_json(report_path)
    report = json.loads(report_text)

    feature_columns = report.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("feature_columns not found in model report")
    return feature_columns


def load_lightgbm_booster(model_path):
    lgb = _import_lightgbm()
    return lgb.Booster(model_file=str(model_path))


def save_predictions(df, output_dir, output_prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{output_prefix}.csv"
    parquet_path = output_dir / f"{output_prefix}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def run_inference(
    input_path=None,
    report_path="data/processed/model_report.json",
    model_dir="data/processed/models",
    output_dir="data/processed",
    output_prefix="inference_predictions",
    last_n=None,
    horizon_steps=0,
    output_suffix=None,
):
    """
    Run inference using saved LightGBM models.

    - Uses point model if present: lightgbm_point.txt
    - Uses quantile models if present: q10/q50/q90
    """
    horizon_steps = int(horizon_steps)
    if horizon_steps < 0:
        raise ValueError("horizon_steps must be >= 0")

    suffix = _format_horizon_suffix(horizon_steps, output_suffix=output_suffix)

    if isinstance(report_path, (str, Path)) and str(report_path) == "data/processed/model_report.json" and suffix:
        report_path = Path("data/processed") / f"model_report{suffix}.json"

    data_path = find_training_file(input_path)
    df = load_table(data_path)

    if "timestamp" not in df.columns:
        raise ValueError("Input dataset must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if last_n is not None:
        if last_n <= 0:
            raise ValueError("last_n must be > 0")
        df = df.tail(last_n).copy()

    feature_columns = parse_feature_columns(report_path)
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns for inference: {missing_features}")

    x = df[feature_columns].to_numpy(dtype=float)

    model_dir = Path(model_dir)
    forecast_delta = pd.Timedelta(minutes=15 * horizon_steps)
    out = pd.DataFrame(
        {
            "timestamp": (df["timestamp"] + forecast_delta).values,
            "source_timestamp": df["timestamp"].values,
        }
    )

    point_model_path = model_dir / f"lightgbm_point{suffix}.txt"
    if point_model_path.exists():
        point_model = load_lightgbm_booster(point_model_path)
        out["lightgbm_pred"] = point_model.predict(x)

    q10_path = model_dir / f"lightgbm_quantile_q10{suffix}.txt"
    q50_path = model_dir / f"lightgbm_quantile_q50{suffix}.txt"
    q90_path = model_dir / f"lightgbm_quantile_q90{suffix}.txt"

    if q10_path.exists() and q50_path.exists() and q90_path.exists():
        out["lightgbm_q10"] = load_lightgbm_booster(q10_path).predict(x)
        out["lightgbm_q50"] = load_lightgbm_booster(q50_path).predict(x)
        out["lightgbm_q90"] = load_lightgbm_booster(q90_path).predict(x)

    if len(out.columns) == 1:
        raise ValueError(
            "No model predictions were produced. Check that LightGBM model files exist in data/processed/models"
        )

    csv_path, parquet_path = save_predictions(out, output_dir=output_dir, output_prefix=f"{output_prefix}{suffix}")
    return out, csv_path, parquet_path


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference from saved LightGBM models")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--report-path", default="data/processed/model_report.json")
    parser.add_argument("--model-dir", default="data/processed/models")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--output-prefix", default="inference_predictions")
    parser.add_argument("--last-n", type=int, default=None, help="Only predict on latest N rows")
    parser.add_argument("--horizon-steps", type=int, default=0, help="Forecast horizon in 15-minute steps (e.g., 4=1h)")
    parser.add_argument("--output-suffix", default=None, help="Optional suffix for output files (e.g., _h1)")
    return parser


def main():
    args = build_parser().parse_args()
    preds, csv_path, parquet_path = run_inference(
        input_path=args.input_path,
        report_path=args.report_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        last_n=args.last_n,
        horizon_steps=args.horizon_steps,
        output_suffix=args.output_suffix,
    )

    print("Inference complete")
    print("Rows predicted:", len(preds))
    print("CSV:", csv_path)
    print("Parquet:", parquet_path)


if __name__ == "__main__":
    main()
