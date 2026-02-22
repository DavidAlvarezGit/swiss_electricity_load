import argparse
from pathlib import Path

from swiss_electricity_load.features import build_training_dataset
from swiss_electricity_load.model import train_models
from swiss_electricity_load.pipeline import run_pipeline
from swiss_electricity_load.predict import run_inference


def parse_lag_steps(text):
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return (1, 4, 96)
    return tuple(int(v) for v in values)


def run_full_flow(
    raw_dir="data/raw",
    processed_dir="data/processed",
    swissgrid_start_year=2009,
    swissgrid_end_year=None,
    weather_start_year=2009,
    target_column=None,
    lag_steps=(1, 4, 96),
    cv_folds=3,
    cv_test_fraction=0.1,
    cv_min_train_fraction=0.4,
    predict_last_n=96,
):
    """
    One-command full flow:
    1) Build data pipeline
    2) Build training features
    3) Train LightGBM models
    4) Run inference
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        swissgrid_start_year=swissgrid_start_year,
        swissgrid_end_year=swissgrid_end_year,
        weather_start_year=weather_start_year,
        skip_weather=False,
    )

    training_path = processed_dir / "training_dataset_quarter_hourly.parquet"
    build_training_dataset(
        input_path=processed_dir,
        output_path=training_path,
        target_column=target_column,
        lag_steps=lag_steps,
        drop_na_rows=True,
    )

    train_models(
        input_path=training_path,
        output_dir=processed_dir,
        target_column=target_column,
        use_lightgbm=True,
        use_lightgbm_quantiles=True,
        cv_folds=cv_folds,
        cv_test_fraction=cv_test_fraction,
        cv_min_train_fraction=cv_min_train_fraction,
        save_models=True,
    )

    predictions, csv_path, parquet_path = run_inference(
        input_path=training_path,
        report_path=processed_dir / "model_report.json",
        model_dir=processed_dir / "models",
        output_dir=processed_dir,
        output_prefix="inference_predictions",
        last_n=predict_last_n,
    )

    return predictions, csv_path, parquet_path


def build_parser():
    parser = argparse.ArgumentParser(description="Run full flow: pipeline -> features -> train -> predict")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--swissgrid-start-year", type=int, default=2009)
    parser.add_argument("--swissgrid-end-year", type=int, default=None)
    parser.add_argument("--weather-start-year", type=int, default=2009)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--lag-steps", default="1,4,96", help="Comma-separated lag steps")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-test-fraction", type=float, default=0.1)
    parser.add_argument("--cv-min-train-fraction", type=float, default=0.4)
    parser.add_argument("--predict-last-n", type=int, default=96)
    return parser


def main():
    args = build_parser().parse_args()
    lag_steps = parse_lag_steps(args.lag_steps)

    preds, csv_path, parquet_path = run_full_flow(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        swissgrid_start_year=args.swissgrid_start_year,
        swissgrid_end_year=args.swissgrid_end_year,
        weather_start_year=args.weather_start_year,
        target_column=args.target_column,
        lag_steps=lag_steps,
        cv_folds=args.cv_folds,
        cv_test_fraction=args.cv_test_fraction,
        cv_min_train_fraction=args.cv_min_train_fraction,
        predict_last_n=args.predict_last_n,
    )

    print("Full flow complete")
    print("Inference rows:", len(preds))
    print("CSV:", csv_path)
    print("Parquet:", parquet_path)


if __name__ == "__main__":
    main()
