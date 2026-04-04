import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Diabetes Progression Risk Dashboard",
    layout="wide"
)

st.title("Diabetes Progression Risk Dashboard")

risk_df = pd.DataFrame(
    {
        "County": ["County A", "County B", "County C", "County D", "County E"],
        "State": ["GA", "NY", "CA", "TX", "GA"],
        "Risk Level": ["High", "High", "Medium", "Medium", "Low"],
        "Diabetes Prevalence": ["12%", "13%", "11%", "10%", "9%"],
        "Risk Score": [90, 80, 70, 60, 50],
        "Year": ["2026", "2026", "2026", "2026", "2026"],
    }
)

county_detail_map = {
    "Fulton County, GA": {
        "risk": "HIGH",
        "prevalence": "12%",
        "factors": [
            "Obesity prevalence",
            "Physical inactivity",
            "Smoking prevalence",
        ],
        "years": [2020, 2021, 2022, 2023],
        "trend": [10, 11, 12, 13],
    },
    "Hinds County, MS": {
        "risk": "HIGH",
        "prevalence": "14%",
        "factors": [
            "Obesity prevalence",
            "Hypertension burden",
            "Limited exercise access",
        ],
        "years": [2020, 2021, 2022, 2023],
        "trend": [11, 12, 13, 14],
    },
    "Jefferson County, AL": {
        "risk": "MEDIUM",
        "prevalence": "10%",
        "factors": [
            "Physical inactivity",
            "Smoking prevalence",
            "Food insecurity",
        ],
        "years": [2020, 2021, 2022, 2023],
        "trend": [8, 9, 10, 10],
    },
}

tab1, tab2 = st.tabs(["Population Risk Overview", "County Detail View"])

with tab1:
    st.header("Population Risk Overview")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filters")

        year = st.selectbox(
            "Year",
            options=["2021", "2022", "2026"],
            index=2
        )

        state = st.selectbox(
            "State",
            options=["All", "Georgia", "New York", "California", "Texas"],
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

        filtered_df = risk_df[risk_df["Year"] == year].copy()

        state_map = {
            "Georgia": "GA",
            "New York": "NY",
            "California": "CA",
            "Texas": "TX",
        }

        if state != "All":
            filtered_df = filtered_df[filtered_df["State"] == state_map[state]]

        if risk_level != "All":
            filtered_df = filtered_df[filtered_df["Risk Level"] == risk_level]

        if county_search.strip():
            filtered_df = filtered_df[
                filtered_df["County"].str.contains(county_search, case=False, na=False)
            ]

        table_df = filtered_df[
            ["County", "State", "Risk Level", "Diabetes Prevalence"]
        ].rename(columns={"Risk Level": "Risk_Score"})

        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.markdown("### Risk Distribution")

        plot_df = filtered_df.sort_values("Risk Score", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(plot_df["County"], plot_df["Risk Score"])
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
            options=list(county_detail_map.keys()),
            index=0
        )

    with col2:
        county_info = county_detail_map[county_detail]

        st.subheader(f"County Detail: {county_detail}")
        st.markdown(f"**Predicted Progression Risk: {county_info['risk']}**")
        st.markdown(
            f"**Current Diabetes Prevalence:** {county_info['prevalence']}"
        )

        inner_col1, inner_col2 = st.columns(2)

        with inner_col1:
            st.markdown("#### Top Contributing Risk Factors")
            for factor in county_info["factors"]:
                st.markdown(f"- {factor}")

        with inner_col2:
            st.markdown("#### Trend Over Time")

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(
                county_info["years"],
                county_info["trend"],
                marker="o"
            )
            ax2.set_title("Diabetes Prevalence by Year")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Diabetes Prevalence (%)")
            plt.tight_layout()

            st.pyplot(fig2)