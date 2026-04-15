
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from DataIngestionMerger_connected import runPipeline, saveOutput

st.set_page_config(
    page_title="Diabetes Progression Risk Dashboard",
    layout="wide"
)

DATA_PATH = "data/processed/county_facts.csv"

RISK_METRICS = {
    "diabetes_pct": "Current diabetes prevalence",
    "obesity_pct": "Obesity prevalence",
    "bphigh_pct": "High blood pressure prevalence",
    "lpa_pct": "Physical inactivity",
    "smoking_pct": "Smoking prevalence",
    "chd_pct": "Coronary heart disease burden",
    "kidney_pct": "Kidney disease burden",
    "sleep_pct": "Insufficient sleep",
    "mhlth_pct": "Poor mental health",
    "diabetes_pct_yoy_change_pos": "Worsening diabetes trend",
}

RISK_WEIGHTS = {
    "diabetes_pct": 0.28,
    "obesity_pct": 0.16,
    "bphigh_pct": 0.14,
    "lpa_pct": 0.12,
    "smoking_pct": 0.10,
    "chd_pct": 0.07,
    "kidney_pct": 0.05,
    "sleep_pct": 0.04,
    "mhlth_pct": 0.04,
    "diabetes_pct_yoy_change_pos": 0.00,  # replaced after derived column added
}

RISK_WEIGHTS["diabetes_pct_yoy_change_pos"] = 1.0 - sum(
    value for key, value in RISK_WEIGHTS.items() if key != "diabetes_pct_yoy_change_pos"
)


def _normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "report_year",
        "totalpopulation",
        "diabetes_pct",
        "obesity_pct",
        "bphigh_pct",
        "lpa_pct",
        "smoking_pct",
        "checkup_pct",
        "chd_pct",
        "kidney_pct",
        "sleep_pct",
        "mhlth_pct",
        "diabetes_pct_lag1",
        "obesity_pct_lag1",
        "diabetes_pct_yoy_change",
        "obesity_pct_yoy_change",
        "diabetes_rank",
        "diabetes_next_year",
        "top_quintile_threshold",
        "target_top_quintile_next_year",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diabetes_pct_yoy_change_pos"] = df["diabetes_pct_yoy_change"].clip(lower=0).fillna(0)
    df["county_display"] = df["locationname"].astype(str) + ", " + df["stateabbr"].astype(str)

    metrics_for_percentile = list(RISK_METRICS.keys())
    for metric in metrics_for_percentile:
        if metric not in df.columns:
            df[metric] = pd.NA

    for metric in metrics_for_percentile:
        pct_col = f"{metric}_pct_rank"
        df[pct_col] = (
            df.groupby("report_year")[metric]
            .rank(method="average", pct=True)
            .fillna(0)
        )

    score = pd.Series(0.0, index=df.index)
    for metric, weight in RISK_WEIGHTS.items():
        score = score + df[f"{metric}_pct_rank"].fillna(0) * weight

    df["risk_score"] = (score * 100).round(1)
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-float("inf"), 33.3333, 66.6666, float("inf")],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    df["diabetes_prevalence_label"] = df["diabetes_pct"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )

    return df


def _prepare_dashboard_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_numeric_columns(df)
    required_cols = {"locationname", "stateabbr", "statedesc", "report_year", "diabetes_pct"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Processed data is missing required columns: {missing_text}")

    df = _add_risk_features(df)
    df = df.sort_values(["county_display", "report_year"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def _read_local_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _build_or_refresh_data(force_refresh: bool = False) -> Tuple[pd.DataFrame, str]:
    if not force_refresh and os.path.exists(DATA_PATH):
        df = _read_local_data(DATA_PATH)
        return _prepare_dashboard_data(df), f"Loaded cached processed data from {DATA_PATH}"

    df = runPipeline()
    saveOutput(df)
    _read_local_data.clear()
    return _prepare_dashboard_data(df), "Fetched fresh CDC PLACES data and rebuilt processed features"


def _top_contributing_factors(row: pd.Series, n: int = 3) -> List[str]:
    scores: Dict[str, float] = {}
    for metric, label in RISK_METRICS.items():
        rank_col = f"{metric}_pct_rank"
        if rank_col in row and pd.notna(row[rank_col]):
            scores[label] = float(row[rank_col])

    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:n]
    return [label for label, _ in top]


def _render_empty_chart(message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    st.pyplot(fig)


st.title("Diabetes Progression Risk Dashboard")
st.caption(
    "This keeps the original dashboard layout, but now reads real county-level CDC PLACES data. "
    "The current risk score is a feature-based ranking proxy built from prevalence and trend inputs "
    "until a trained prediction model is added."
)

top_col_left, top_col_right = st.columns([1, 4])

with top_col_left:
    refresh_clicked = st.button("Refresh Data", use_container_width=True)

try:
    dashboard_df, data_status = _build_or_refresh_data(force_refresh=refresh_clicked)
except Exception as exc:
    st.error(
        "The dashboard could not load backend data. "
        "Either place a processed file at data/processed/county_facts.csv "
        "or allow the app to reach the CDC API."
    )
    st.exception(exc)
    st.stop()

with top_col_right:
    st.info(data_status)

years = sorted(dashboard_df["report_year"].dropna().astype(int).unique().tolist())
states = sorted(dashboard_df["statedesc"].dropna().astype(str).unique().tolist())
county_options = sorted(dashboard_df["county_display"].dropna().astype(str).unique().tolist())

default_year = years[-1] if years else None

tab1, tab2 = st.tabs(["Population Risk Overview", "County Detail View"])

with tab1:
    st.header("Population Risk Overview")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filters")

        year = st.selectbox(
            "Year",
            options=years,
            index=len(years) - 1 if years else 0
        )

        state = st.selectbox(
            "State",
            options=["All"] + states,
            index=0
        )

        risk_level = st.selectbox(
            "Risk Level",
            options=["All", "High", "Medium", "Low"],
            index=0
        )

        county_search = st.text_input(
            "Search County",
            placeholder="Enter county name"
        )

    with col2:
        st.subheader("Top Counties by Predicted Diabetes Risk")

        filtered_df = dashboard_df[dashboard_df["report_year"] == year].copy()

        if state != "All":
            filtered_df = filtered_df[filtered_df["statedesc"] == state]

        if risk_level != "All":
            filtered_df = filtered_df[filtered_df["risk_level"] == risk_level]

        if county_search.strip():
            filtered_df = filtered_df[
                filtered_df["locationname"].str.contains(county_search, case=False, na=False)
            ]

        filtered_df = filtered_df.sort_values("risk_score", ascending=False)

        table_df = filtered_df[
            ["locationname", "stateabbr", "risk_level", "diabetes_prevalence_label"]
        ].rename(columns={
            "locationname": "County",
            "stateabbr": "State",
            "risk_level": "Risk Level",
            "diabetes_prevalence_label": "Diabetes Prevalence",
        })

        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.markdown("### Risk Distribution")

        if filtered_df.empty:
            _render_empty_chart("No counties match the current filters.")
        else:
            plot_df = filtered_df.head(25)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(plot_df["locationname"], plot_df["risk_score"])
            ax.set_title("Counties with Highest Predicted Risk")
            ax.set_ylabel("Risk Score")
            ax.set_xlabel("County")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            st.pyplot(fig)

with tab2:
    st.header("County Detail View")

    col1, col2 = st.columns([1, 3])

    with col1:
        county_detail = st.selectbox(
            "Select County",
            options=county_options,
            index=0 if county_options else None
        )

    with col2:
        county_rows = dashboard_df[dashboard_df["county_display"] == county_detail].copy()
        county_rows = county_rows.sort_values("report_year")

        if county_rows.empty:
            st.warning("No county detail is available for the selected county.")
        else:
            latest_row = county_rows.iloc[-1]
            factors = _top_contributing_factors(latest_row)

            st.subheader(f"County Detail: {county_detail}")
            st.markdown(f"**Predicted Progression Risk: {str(latest_row['risk_level']).upper()}**")
            st.markdown(
                f"**Current Diabetes Prevalence:** {latest_row['diabetes_prevalence_label']}"
            )

            inner_col1, inner_col2 = st.columns(2)

            with inner_col1:
                st.markdown("#### Top Contributing Risk Factors")
                for factor in factors:
                    st.markdown(f"- {factor}")

            with inner_col2:
                st.markdown("#### Trend Over Time")

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(
                    county_rows["report_year"],
                    county_rows["diabetes_pct"],
                    marker="o"
                )
                ax2.set_title("Diabetes Prevalence by Year")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Diabetes Prevalence (%)")
                plt.tight_layout()

                st.pyplot(fig2)
