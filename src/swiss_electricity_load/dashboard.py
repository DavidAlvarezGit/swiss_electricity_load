from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime, timezone


def _safe_import_streamlit():
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Streamlit is not installed. Install with: poetry add streamlit\n"
            "Then run: poetry run swiss-load-dashboard"
        ) from exc
    return st


def _cache_data(func):
    st = _safe_import_streamlit()
    return st.cache_data(show_spinner=False)(func)


@_cache_data
def load_table(base_dir, stem, suffix=""):
    base_dir = Path(base_dir)
    parquet_path = base_dir / f"{stem}{suffix}.parquet"
    csv_path = base_dir / f"{stem}{suffix}.csv"

    if parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(parquet_path), parquet_path
    if csv_path.exists():
        import pandas as pd

        return pd.read_csv(csv_path), csv_path
    return None, None


@_cache_data
def load_report(processed_dir, suffix=""):
    report_path = Path(processed_dir) / f"model_report{suffix}.json"
    if not report_path.exists():
        return None, None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report, report_path


def _format_number(value, decimals=2):
    if value is None:
        return "-"
    return f"{value:,.{decimals}f}"


def _improvement_pct(reference, candidate):
    if reference in (None, 0) or candidate is None:
        return None
    return (reference - candidate) / reference * 100


def _to_time_sorted(df):
    import pandas as pd

    if df is None or "timestamp" not in df.columns:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _available_years(df):
    import pandas as pd

    if df is None or "timestamp" not in df.columns:
        return []
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    years = ts.dt.year.dropna().unique().tolist()
    return sorted(int(y) for y in years)


def _filter_by_year(df, year):
    import pandas as pd

    if df is None or year in (None, "All"):
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.loc[ts.dt.year == int(year)].copy()


def _build_metric_table(report):
    import pandas as pd

    return pd.DataFrame(
        [
            {"model": "baseline", **(report.get("baseline_metrics") or {})},
            {"model": "linear", **(report.get("linear_metrics") or {})},
            {"model": "lightgbm", **(report.get("lightgbm_metrics") or {})},
        ]
    )


def _build_cv_table(cv):
    import pandas as pd

    rows = []
    for row in cv.get("per_fold", []):
        rows.append(
            {
                "fold": row.get("fold"),
                "n_train": row.get("n_train"),
                "n_test": row.get("n_test"),
                "baseline_mae": (row.get("baseline_metrics") or {}).get("mae"),
                "linear_mae": (row.get("linear_metrics") or {}).get("mae"),
                "lightgbm_mae": (row.get("lightgbm_metrics") or {}).get("mae") if row.get("lightgbm_metrics") else None,
            }
        )
    return pd.DataFrame(rows)


def _build_artifact_table(processed_dir):
    import pandas as pd

    rows = []
    for p in sorted(Path(processed_dir).glob("*")):
        if p.is_file():
            rows.append({"name": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 2), "path": str(p)})
    return pd.DataFrame(rows)


def _downsample_time_df(df, max_points):
    if df is None or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def render_timeseries_chart(st, df, y_columns, title=None, x_col="timestamp", x_is_time=True):
    import altair as alt
    import pandas as pd

    if df is None or df.empty or not y_columns:
        st.info("No data to plot.")
        return

    use_cols = [x_col] + [c for c in y_columns if c in df.columns]
    if len(use_cols) <= 1:
        st.info("No selected series available in data.")
        return

    plot_df = df[use_cols].copy()
    if x_is_time:
        plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col])

    long_df = plot_df.melt(id_vars=[x_col], var_name="series", value_name="value")

    chart = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{x_col}:{'T' if x_is_time else 'Q'}", title=("Timestamp" if x_is_time else x_col.capitalize())),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color(
                "series:N",
                title="Series",
                scale=alt.Scale(
                    range=[
                        "#1f77b4",  # blue
                        "#d62728",  # red
                        "#2ca02c",  # green
                        "#ff7f0e",  # orange
                        "#9467bd",  # purple
                        "#17becf",  # cyan
                    ]
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=",.2f"),
            ],
        )
        .properties(height=420, title=title)
        .interactive()
        .configure(background="#ffffff")
        .configure_axis(
            labelColor="#0f172a",
            titleColor="#0f172a",
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#94a3b8",
        )
        .configure_title(color="#0f172a", fontSize=18, fontWeight="bold")
        .configure_legend(labelColor="#0f172a", titleColor="#0f172a")
        .configure_view(strokeWidth=0)
    )

    # Disable Streamlit theme override to keep charts high-contrast and readable.
    st.altair_chart(chart, width="stretch", theme=None)


def render_dashboard(processed_dir="data/processed"):
    st = _safe_import_streamlit()
    import pandas as pd

    st.set_page_config(page_title="Swiss Load Dashboard", page_icon="dashboard", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #ffffff; }
        [data-testid="stSidebar"] { background: #f3f4f6; }
        .stMarkdown, .stCaption, .stText, .stMetric, label, p, h1, h2, h3, h4, h5, h6, span, div { color: #111827; }
        [data-baseweb="select"] * { color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    processed_dir = Path(processed_dir)

    st.sidebar.header("Controls")
    processed_dir = Path(st.sidebar.text_input("Processed directory", str(processed_dir)))
    chart_points = st.sidebar.slider("Max chart points", min_value=200, max_value=5000, value=1000, step=100)
    show_all_points = st.sidebar.checkbox("Show all points (no downsample)", value=True)
    show_tail_rows = st.sidebar.slider("Table rows", min_value=20, max_value=300, value=60, step=20)
    show_heavy_tables = st.sidebar.checkbox("Show large data tables", value=False)
    force_reload = st.sidebar.button("Reload data")
    if force_reload:
        st.cache_data.clear()
    st.sidebar.caption(f"Refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    st.title("Swiss Electricity Load Forecasting Dashboard")
    st.caption("Professional model monitoring for training, validation, and inference artifacts.")

    def render_forecast_panel(horizon_suffix, horizon_label):
        report, report_path = load_report(processed_dir, suffix=horizon_suffix)
        preds, preds_path = load_table(processed_dir, "model_predictions", suffix=horizon_suffix)
        inf, inf_path = load_table(processed_dir, "inference_predictions", suffix=horizon_suffix)

        if report is None:
            st.error(f"model_report{horizon_suffix}.json not found. Train with --horizon-steps for {horizon_label}.")
            return

        years = sorted(set(_available_years(preds) + _available_years(inf)))
        year_options = ["All"] + [str(y) for y in years]
        year_choice = st.selectbox("Year filter", options=year_options, key=f"year_filter_{horizon_label}")
        year_filter = None if year_choice == "All" else int(year_choice)

        baseline_mae = report.get("baseline_metrics", {}).get("mae")
        linear_mae = report.get("linear_metrics", {}).get("mae")
        lgbm_metrics = report.get("lightgbm_metrics")
        lgbm_mae = lgbm_metrics.get("mae") if lgbm_metrics else None

        linear_gain = _improvement_pct(baseline_mae, linear_mae)
        lgbm_gain = _improvement_pct(baseline_mae, lgbm_mae)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows (Train)", f"{report.get('n_rows_train', 0):,}")
        c2.metric("Rows (Test)", f"{report.get('n_rows_test', 0):,}")
        c3.metric("Baseline MAE", _format_number(baseline_mae))
        c4.metric("Linear MAE", _format_number(linear_mae), delta=(f"{linear_gain:.2f}% vs baseline" if linear_gain is not None else None))
        c5.metric("LightGBM MAE", _format_number(lgbm_mae), delta=(f"{lgbm_gain:.2f}% vs baseline" if lgbm_gain is not None else None))
        if lgbm_metrics is None:
            horizon_steps = report.get("horizon_steps", None)
            if horizon_steps is None:
                st.info("LightGBM metrics not found in report. Train with `--use-lightgbm` to populate.")
            else:
                st.info(f"LightGBM metrics not found. Train with `swiss-load-train --use-lightgbm --horizon-steps {horizon_steps}`.")

        tabs = st.tabs(["Performance", "Predictions", "Cross-Validation", "Inference"])

        with tabs[0]:
            st.subheader("Model Performance Summary")
            table = _build_metric_table(report)
            st.dataframe(table, width="stretch")

        with tabs[1]:
            st.subheader("Test Predictions")
            if preds is None:
                st.info("model_predictions file not found.")
            else:
                preds = _to_time_sorted(preds)
                preds = _filter_by_year(preds, year_filter)

                if preds is None or preds.empty:
                    st.warning("Prediction table is empty.")
                else:
                    view = preds.copy()
                    if not show_all_points:
                        view = _downsample_time_df(view, chart_points)

                    all_pred_cols = [c for c in ["baseline_pred", "linear_pred", "lightgbm_pred"] if c in view.columns]
                    selected_pred_cols = st.multiselect(
                        "Prediction series",
                        options=all_pred_cols,
                        default=all_pred_cols,
                        key=f"pred_series_{horizon_label}",
                    )
                    plot_mode = st.radio(
                        "Plot mode",
                        options=["Actual scale", "Indexed (start=100)", "Deviation vs y_true"],
                        horizontal=True,
                        key=f"plot_mode_{horizon_label}",
                    )

                    if "y_true" in view.columns and selected_pred_cols:
                        if plot_mode == "Actual scale":
                            line_cols = ["y_true"] + selected_pred_cols
                            render_timeseries_chart(st, view, line_cols, title="Predictions vs Actual")
                        elif plot_mode == "Indexed (start=100)":
                            idx_df = view[["timestamp", "y_true"] + selected_pred_cols].copy()
                            for col in ["y_true"] + selected_pred_cols:
                                base = idx_df[col].iloc[0]
                                if base == 0:
                                    idx_df[col] = 0.0
                                else:
                                    idx_df[col] = idx_df[col] / base * 100.0
                            st.caption("Each series is re-scaled to 100 at the first displayed timestamp.")
                            render_timeseries_chart(st, idx_df, [c for c in idx_df.columns if c != "timestamp"], title="Indexed Series (Start=100)")
                        else:
                            dev_df = view[["timestamp"]].copy()
                            for col in selected_pred_cols:
                                dev_df[f"{col}_minus_y_true"] = view[col] - view["y_true"]
                            st.caption("Deviation chart: prediction - y_true (closer to 0 is better).")
                            render_timeseries_chart(st, dev_df, [c for c in dev_df.columns if c != "timestamp"], title="Deviation vs y_true")

                    residual_model = st.selectbox(
                        "Residual diagnostics model",
                        options=all_pred_cols,
                        index=min(1, len(all_pred_cols) - 1),
                        key=f"residual_model_{horizon_label}",
                    )
                    if residual_model and "y_true" in view.columns:
                        residual_col = "residual"
                        view[residual_col] = view["y_true"] - view[residual_model]
                        st.caption(f"Residuals = y_true - {residual_model}")
                        st.line_chart(view.set_index("timestamp")[[residual_col]])

                    if {"lightgbm_q10", "lightgbm_q50", "lightgbm_q90", "y_true"}.issubset(view.columns):
                        q_cols = ["lightgbm_q10", "lightgbm_q50", "lightgbm_q90", "y_true"]
                        render_timeseries_chart(st, view, q_cols, title="Quantile Bands and Actual")
                        coverage = ((view["y_true"] >= view["lightgbm_q10"]) & (view["y_true"] <= view["lightgbm_q90"])).mean() * 100
                        width = (view["lightgbm_q90"] - view["lightgbm_q10"]).mean()
                        qc1, qc2 = st.columns(2)
                        qc1.metric("Quantile Coverage (q10-q90)", f"{coverage:.2f}%")
                        qc2.metric("Average Interval Width", _format_number(width))

                    if show_heavy_tables:
                        st.dataframe(view.tail(show_tail_rows), width="stretch")
                    else:
                        st.caption("Table hidden for performance. Enable 'Show large data tables' in sidebar.")

        with tabs[2]:
            st.subheader("Time-Series CV")
            cv = report.get("time_series_cv")
            if not cv:
                st.info("No CV results found in report. Train with --cv-folds >= 2")
            else:
                st.write(f"Folds used: **{cv.get('n_folds')}**")
                summary = pd.DataFrame(
                    [
                        {"model": "baseline", **(cv.get("mean_baseline_metrics") or {})},
                        {"model": "linear", **(cv.get("mean_linear_metrics") or {})},
                        {"model": "lightgbm", **(cv.get("mean_lightgbm_metrics") or {})},
                    ]
                )
                st.dataframe(summary, width="stretch")

                per_fold_df = _build_cv_table(cv)
                if not per_fold_df.empty:
                    st.dataframe(per_fold_df, width="stretch")
                    chart_cols = [c for c in ["baseline_mae", "linear_mae", "lightgbm_mae"] if c in per_fold_df.columns]
                    if chart_cols:
                        render_timeseries_chart(st, per_fold_df, chart_cols, title="CV Metric by Fold", x_col="fold", x_is_time=False)

        with tabs[3]:
            st.subheader("Latest Inference")
            if inf is None:
                st.info("No inference_predictions file found. Run swiss-load-predict or swiss-load-fullflow.")
            else:
                inf = _to_time_sorted(inf)
                inf = _filter_by_year(inf, year_filter)
                if inf is None or inf.empty:
                    st.warning("Inference table is empty.")
                else:
                    inf_view = inf.copy()
                    if not show_all_points:
                        inf_view = _downsample_time_df(inf_view, chart_points)
                    cols = [c for c in ["lightgbm_pred", "lightgbm_q10", "lightgbm_q50", "lightgbm_q90"] if c in inf_view.columns]
                    if cols:
                        render_timeseries_chart(st, inf_view, cols, title="Latest Inference")
                    if show_heavy_tables:
                        st.dataframe(inf_view.tail(show_tail_rows), width="stretch")
                    else:
                        st.caption("Table hidden for performance. Enable 'Show large data tables' in sidebar.")

                    csv_bytes = inf_view.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Displayed Inference CSV",
                        data=csv_bytes,
                        file_name=f"inference_subset{horizon_suffix}.csv",
                        mime="text/csv",
                        key=f"download_{horizon_label}",
                    )

        st.divider()
        st.subheader("Artifacts")
        st.write(f"Processed directory: `{processed_dir}`")
        if report_path:
            st.write(f"Report: `{report_path}`")
        if preds_path:
            st.write(f"Predictions: `{preds_path}`")
        if inf_path:
            st.write(f"Inference: `{inf_path}`")

        saved_models = report.get("saved_models", [])
        if saved_models:
            st.write("Saved models:")
            for m in saved_models:
                st.write(f"- `{m}`")

    horizon_tabs = st.tabs(["Forecast 1h", "Forecast 12h", "Artifacts"])

    with horizon_tabs[0]:
        render_forecast_panel("_h1", "1h")

    with horizon_tabs[1]:
        render_forecast_panel("_h12", "12h")

    with horizon_tabs[2]:
        st.subheader("Artifacts")
        st.write(f"Processed directory: `{processed_dir}`")
        art_df = _build_artifact_table(processed_dir)
        if not art_df.empty:
            st.dataframe(art_df, width="stretch")


def launch():
    """Launch streamlit app from CLI script."""
    app_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    raise SystemExit(subprocess.call(cmd))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Swiss load Streamlit dashboard")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    render_dashboard(processed_dir=args.processed_dir)


if __name__ == "__main__":
    main()
