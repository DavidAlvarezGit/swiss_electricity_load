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


@_cache_data
def load_lightgbm_feature_importance(processed_dir, suffix="_h24"):
    import pandas as pd
    import re

    report, _ = load_report(processed_dir, suffix=suffix)
    if report is None:
        return None

    model_path = Path(processed_dir) / "models" / f"lightgbm_point{suffix}.txt"
    if not model_path.exists():
        return None

    try:
        import lightgbm as lgb
    except Exception:
        return None

    booster = lgb.Booster(model_file=str(model_path))
    feature_names = report.get("feature_columns", []) or booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(len(gain))]

    n = min(len(feature_names), len(gain), len(split))
    if n == 0:
        return None

    imp = pd.DataFrame(
        {
            "feature": feature_names[:n],
            "gain": gain[:n],
            "split": split[:n],
        }
    )
    imp = imp[imp["gain"] > 0].copy()
    if imp.empty:
        return None

    rename_map = {
        "hour": "Hour of Day",
        "minute": "Minute of Hour",
        "day_of_week": "Day of Week",
        "day_of_month": "Day of Month",
        "month": "Month",
        "is_weekend": "Weekend Flag",
        "temp_weighted": "Weighted Temperature",
        "temp_72h": "72h Avg Temperature",
        "HDH": "Heating Degree Hours",
        "CDH": "Cooling Degree Hours",
        "extreme_cold": "Extreme Cold Indicator",
        "zurich": "Zurich Temperature",
        "geneva": "Geneva Temperature",
        "basel": "Basel Temperature",
        "bern": "Bern Temperature",
        "lausanne": "Lausanne Temperature",
        "lugano": "Lugano Temperature",
    }

    def _feature_label(name):
        text = str(name)
        m = re.search(r"_lag_(\d+)", text)
        if m:
            steps = int(m.group(1))
            minutes = steps * 15
            if minutes % 60 == 0:
                return f"Load Lag (t-{minutes // 60}h)"
            return f"Load Lag (t-{minutes}m)"
        return rename_map.get(text, text.replace("_", " ").title())

    imp["feature"] = imp["feature"].map(_feature_label)
    imp["gain_pct"] = imp["gain"] / imp["gain"].sum() * 100.0
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


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


def _available_years(df, timestamp_col="timestamp"):
    import pandas as pd

    if df is None or timestamp_col not in df.columns:
        return []
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    years = ts.dt.year.dropna().unique().tolist()
    return sorted(int(y) for y in years)


def _filter_by_year(df, year, timestamp_col="timestamp"):
    import pandas as pd

    if df is None or year in (None, "All"):
        return df
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
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


def _downsample_time_df(df, max_points, every_n=None):
    if df is None or len(df) <= max_points:
        return df
    if every_n is not None and every_n > 1:
        step = every_n
    else:
        step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def render_timeseries_chart(st, df, y_columns, title=None, x_col="timestamp", x_is_time=True, series_label_map=None):
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
    if series_label_map:
        long_df["series"] = long_df["series"].map(lambda s: series_label_map.get(s, s))

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


def render_quantile_band_chart(st, df, title=None, x_col="timestamp", include_actual=True):
    import altair as alt
    import pandas as pd

    required = {"lightgbm_q10", "lightgbm_q50", "lightgbm_q90"}
    if df is None or df.empty or not required.issubset(df.columns):
        st.info("Quantile band data is not available.")
        return

    use_cols = [x_col, "lightgbm_q10", "lightgbm_q50", "lightgbm_q90"]
    if include_actual and "y_true" in df.columns:
        use_cols.append("y_true")

    plot_df = df[use_cols].copy()
    plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col]).reset_index(drop=True)
    if plot_df.empty:
        st.info("No data to plot.")
        return

    band = (
        alt.Chart(plot_df)
        .mark_area(opacity=0.22, color="#1d4ed8")
        .encode(
            x=alt.X(f"{x_col}:T", title="Timestamp"),
            y=alt.Y("lightgbm_q10:Q", title="Load"),
            y2="lightgbm_q90:Q",
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title="Time"),
                alt.Tooltip("lightgbm_q10:Q", title="P10", format=",.2f"),
                alt.Tooltip("lightgbm_q50:Q", title="P50", format=",.2f"),
                alt.Tooltip("lightgbm_q90:Q", title="P90", format=",.2f"),
            ],
        )
    )

    p50 = (
        alt.Chart(plot_df)
        .mark_line(color="#1d4ed8", strokeWidth=2.5)
        .encode(x=f"{x_col}:T", y="lightgbm_q50:Q")
    )

    q10_line = (
        alt.Chart(plot_df)
        .mark_line(color="#60a5fa", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(x=f"{x_col}:T", y="lightgbm_q10:Q")
    )
    q90_line = (
        alt.Chart(plot_df)
        .mark_line(color="#60a5fa", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(x=f"{x_col}:T", y="lightgbm_q90:Q")
    )

    chart = band + q10_line + q90_line + p50

    if include_actual and "y_true" in plot_df.columns:
        actual = (
            alt.Chart(plot_df)
            .mark_line(color="#0f172a", strokeWidth=2)
            .encode(x=f"{x_col}:T", y="y_true:Q")
        )
        chart = chart + actual

    chart = (
        chart.properties(height=420, title=title)
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
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, width="stretch", theme=None)


def render_feature_importance_chart(st, imp_df, top_n=15):
    import altair as alt

    if imp_df is None or imp_df.empty:
        st.info("Feature importance is not available.")
        return

    view = imp_df.head(top_n).copy()

    chart = (
        alt.Chart(view)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            x=alt.X("gain_pct:Q", title="Importance (% of total gain)"),
            color=alt.Color(
                "gain_pct:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("gain_pct:Q", title="Importance %", format=".2f"),
                alt.Tooltip("split:Q", title="Splits", format=",.0f"),
            ],
        )
        .properties(height=420, title="Top LightGBM Drivers")
        .configure(background="#ffffff")
        .configure_axis(
            labelColor="#0f172a",
            titleColor="#0f172a",
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#94a3b8",
        )
        .configure_title(color="#0f172a", fontSize=18, fontWeight="bold")
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, width="stretch", theme=None)


def render_dashboard(processed_dir="data/processed"):
    st = _safe_import_streamlit()
    import pandas as pd

    st.set_page_config(page_title="Swiss Load Dashboard", page_icon="dashboard", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #fbfbfd; }
        [data-testid="stSidebar"] { background: #f4f6fb; }
        .stMarkdown, .stCaption, .stText, .stMetric, label, p, h1, h2, h3, h4, h5, h6, span, div { color: #0f172a; }
        [data-baseweb="select"] * { color: #0f172a; }
        [data-baseweb="popover"] * { color: #0f172a; }
        [data-baseweb="menu"] { background: #ffffff; }
        [data-baseweb="menu"] * { color: #0f172a !important; }
        [role="listbox"] { background: #ffffff !important; }
        [role="listbox"] * { color: #0f172a !important; }
        [role="option"] { background: #ffffff !important; color: #0f172a !important; }
        [role="option"][aria-selected="true"] { background: #e5e7eb !important; }
        [data-baseweb="select"] > div { background: #ffffff; border: 1px solid #cbd5e1; }
        input, textarea { background: #ffffff !important; color: #0f172a !important; }
        [data-testid="stAlert"] {
            background: #dbeafe !important;
            color: #0f172a !important;
            border: 1px solid #93c5fd !important;
        }
        [data-testid="stAlert"] * { color: #0f172a !important; }
        .stButton > button, .stDownloadButton > button {
            background: #1d4ed8 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #1e40af !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    processed_dir = Path(processed_dir)

    st.sidebar.header("Controls")
    processed_dir = Path(st.sidebar.text_input("Processed directory", str(processed_dir)))
    chart_points = st.sidebar.slider("Max chart points", min_value=100, max_value=2000, value=300, step=50)
    downsample_mode = "Auto"
    every_n = None
    show_tail_rows = 0
    force_reload = st.sidebar.button("Reload data")
    if force_reload:
        st.cache_data.clear()
    st.sidebar.caption(f"Refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    st.title("Swiss Electricity Demand Outlook (24h)")
    st.caption("Executive view focused on next-day demand, forecast performance, and model drivers.")

    def render_forecast_panel(horizon_suffix, horizon_label):
        report, _ = load_report(processed_dir, suffix=horizon_suffix)
        preds, _ = load_table(processed_dir, "model_predictions", suffix=horizon_suffix)
        inf, _ = load_table(processed_dir, "inference_predictions", suffix=horizon_suffix)

        if report is None:
            st.error(f"model_report{horizon_suffix}.json not found. Train with --horizon-steps for {horizon_label}.")
            return

        filter_col = "source_timestamp" if (preds is not None and "source_timestamp" in preds.columns) else "timestamp"
        if inf is not None and "source_timestamp" in inf.columns:
            filter_col = "source_timestamp"

        year_filter = None

        baseline_mae = report.get("baseline_metrics", {}).get("mae")
        lgbm_metrics = report.get("lightgbm_metrics")
        lgbm_mae = lgbm_metrics.get("mae") if lgbm_metrics else None

        lgbm_gain = _improvement_pct(baseline_mae, lgbm_mae)

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline MAE", _format_number(baseline_mae))
        c2.metric("LightGBM MAE", _format_number(lgbm_mae))
        c3.metric(
            "LightGBM Improvement",
            (f"{lgbm_gain:.2f}%" if lgbm_gain is not None else "-"),
            delta=("vs baseline" if lgbm_gain is not None else None),
        )
        if lgbm_metrics is None:
            horizon_steps = report.get("horizon_steps", None)
            if horizon_steps is None:
                st.info("LightGBM metrics not found in report. Train with `--use-lightgbm` to populate.")
            else:
                st.info(f"LightGBM metrics not found. Train with `swiss-load-train --use-lightgbm --horizon-steps {horizon_steps}`.")

        tabs = st.tabs(["Predictions", "Inference", "Model Drivers"])

        with tabs[0]:
            st.subheader("Test Predictions")
            if preds is None:
                st.info("model_predictions file not found.")
            else:
                preds = _to_time_sorted(preds)
                preds = _filter_by_year(preds, year_filter, filter_col)

                if preds is None or preds.empty:
                    st.warning("Prediction table is empty.")
                else:
                    view = preds.copy()
                    view = _downsample_time_df(view, chart_points, every_n=every_n)

                    all_pred_cols = [c for c in ["baseline_pred", "lightgbm_pred"] if c in view.columns]
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
                            render_timeseries_chart(
                                st,
                                view,
                                line_cols,
                                title="Predictions vs Actual",
                                series_label_map={
                                    "y_true": "Actual",
                                    "baseline_pred": "Baseline",
                                    "lightgbm_pred": "LightGBM",
                                },
                            )
                        elif plot_mode == "Indexed (start=100)":
                            idx_df = view[["timestamp", "y_true"] + selected_pred_cols].copy()
                            for col in ["y_true"] + selected_pred_cols:
                                base = idx_df[col].iloc[0]
                                if base == 0:
                                    idx_df[col] = 0.0
                                else:
                                    idx_df[col] = idx_df[col] / base * 100.0
                            st.caption("Each series is re-scaled to 100 at the first displayed timestamp.")
                            render_timeseries_chart(
                                st,
                                idx_df,
                                [c for c in idx_df.columns if c != "timestamp"],
                                title="Indexed Series (Start=100)",
                                series_label_map={
                                    "y_true": "Actual",
                                    "baseline_pred": "Baseline",
                                    "lightgbm_pred": "LightGBM",
                                },
                            )
                        else:
                            dev_df = view[["timestamp"]].copy()
                            for col in selected_pred_cols:
                                dev_df[f"{col}_minus_y_true"] = view[col] - view["y_true"]
                            st.caption("Deviation chart: prediction - y_true (closer to 0 is better).")
                            render_timeseries_chart(
                                st,
                                dev_df,
                                [c for c in dev_df.columns if c != "timestamp"],
                                title="Deviation vs y_true",
                                series_label_map={
                                    "baseline_pred_minus_y_true": "Baseline Error",
                                    "lightgbm_pred_minus_y_true": "LightGBM Error",
                                },
                            )

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
                        render_quantile_band_chart(st, view, title="LightGBM Prediction Interval (P10-P90)", include_actual=True)
                        coverage = ((view["y_true"] >= view["lightgbm_q10"]) & (view["y_true"] <= view["lightgbm_q90"])).mean() * 100
                        width = (view["lightgbm_q90"] - view["lightgbm_q10"]).mean()
                        qc1, qc2 = st.columns(2)
                        qc1.metric("Quantile Coverage (q10-q90)", f"{coverage:.2f}%")
                        qc2.metric("Average Interval Width", _format_number(width))

                    st.caption("Table hidden for performance.")

        with tabs[1]:
            st.subheader("Latest Inference")
            if inf is None:
                st.info("No inference_predictions file found. Run swiss-load-predict or swiss-load-fullflow.")
            else:
                inf = _to_time_sorted(inf)
                inf = _filter_by_year(inf, year_filter, filter_col)
                if inf is None or inf.empty:
                    st.warning("Inference table is empty.")
                else:
                    inf_view = inf.copy()
                    inf_view = _downsample_time_df(inf_view, chart_points, every_n=every_n)
                    cols = [c for c in ["lightgbm_pred", "lightgbm_q10", "lightgbm_q50", "lightgbm_q90"] if c in inf_view.columns]
                    if {"lightgbm_q10", "lightgbm_q50", "lightgbm_q90"}.issubset(inf_view.columns):
                        render_quantile_band_chart(
                            st,
                            inf_view,
                            title="Latest Inference Interval (P10-P90)",
                            include_actual=False,
                        )
                    elif cols:
                        render_timeseries_chart(st, inf_view, cols, title="Latest Inference")
                    st.caption("Table hidden for performance.")

                    csv_bytes = inf_view.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Displayed Inference CSV",
                        data=csv_bytes,
                        file_name=f"inference_subset{horizon_suffix}.csv",
                        mime="text/csv",
                        key=f"download_{horizon_label}",
                    )

        with tabs[2]:
            st.subheader("Variable Importance")
            imp = load_lightgbm_feature_importance(processed_dir, suffix=horizon_suffix)
            if imp is None:
                st.info("LightGBM model importance not available. Train with `--use-lightgbm` for this horizon.")
            else:
                render_feature_importance_chart(st, imp, top_n=15)
                st.caption("Importance is based on total LightGBM gain (higher means stronger contribution).")
                top5 = float(imp["gain_pct"].head(5).sum())
                st.metric("Top-5 Driver Share", f"{top5:.2f}%")
                st.dataframe(
                    imp[["feature", "gain_pct", "split"]].rename(
                        columns={"gain_pct": "importance_pct", "split": "split_count"}
                    ),
                    width="stretch",
                )

    render_forecast_panel("_h24", "24h")

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
