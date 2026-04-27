import pandas as pd
import os

# -------------------------------------------------------
# CDC PLACES Ingestion + Feature Engineering
# Multi-dataset version
# -------------------------------------------------------

# --- STEP 1: Load raw data (MULTIPLE DATASETS) ---

# TODO: Add additional dataset file paths here as you download them
FILE_PATHS = [
    "/PLACES__Local_Data_for_Better_Health,_County_Data,_2025_release_20260322.csv",
    # "/PLACES_2024.csv",
    # "/PLACES_2023.csv",
    # "/PLACES_2022.csv",
]

dfs = []

for path in FILE_PATHS:
    try:
        df_temp = pd.read_csv(path)
        print(f"Loaded {path}: {df_temp.shape}")
        dfs.append(df_temp)
    except Exception as e:
        print(f"Failed to load {path}: {e}")

# Combine all datasets into one DataFrame
df = pd.concat(dfs, ignore_index=True)

print(f"Combined dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# NOTE:
# If multiple datasets contain overlapping years/counties,
# newer datasets should generally be more accurate.
# If needed, we can later deduplicate by keeping the newest values. I have code
# that ought to work for that commented out in the DataIngestionMerger.py file

# -------------------------------------------------------
# --- STEP 2: Filter rows ---
# -------------------------------------------------------

MEASURE_IDS = [
    "DIABETES", "OBESITY", "BPHIGH", "LPA",
    "CSMOKING", "CHECKUP", "CHD", "KIDNEY",
    "SLEEP", "MHLTH"
]

VALUE_TYPE = "CrdPrv"

df_filtered = df[
    (df["MeasureId"].isin(MEASURE_IDS)) &
    (df["DataValueTypeID"] == VALUE_TYPE)
]

print(f"After filtering: {df_filtered.shape[0]} rows")

# --- STEP 2b: Keep only columns we need ---
COLUMNS_TO_KEEP = [
    "Year", "StateAbbr", "StateDesc",
    "LocationName", "LocationID",
    "MeasureId", "Data_Value",
    "Low_Confidence_Limit", "High_Confidence_Limit",
    "TotalPopulation", "Geolocation"
]

df_filtered = df_filtered[COLUMNS_TO_KEEP]

# -------------------------------------------------------
# --- STEP 3: Pivot to wide format ---
# -------------------------------------------------------

df_wide = df_filtered.pivot_table(
    index=[
        "Year", "StateAbbr", "StateDesc",
        "LocationName", "LocationID",
        "TotalPopulation", "Geolocation"
    ],
    columns="MeasureId",
    values="Data_Value"
).reset_index()

df_wide.columns.name = None

df_wide = df_wide.rename(columns={
    "DIABETES":  "diabetes_pct",
    "OBESITY":   "obesity_pct",
    "BPHIGH":    "bphigh_pct",
    "LPA":       "lpa_pct",
    "CSMOKING":  "smoking_pct",
    "CHECKUP":   "checkup_pct",
    "CHD":       "chd_pct",
    "KIDNEY":    "kidney_pct",
    "SLEEP":     "sleep_pct",
    "MHLTH":     "mhlth_pct",
    "LocationID": "fips_code",
    "Year":       "report_year"
})

df_wide["county_year_id"] = (
    df_wide["fips_code"].astype(str) + "_" +
    df_wide["report_year"].astype(str)
)

print(f"Wide format: {df_wide.shape}")

# -------------------------------------------------------
# --- STEP 4: FEATURE ENGINEERING ---
# -------------------------------------------------------

df_wide = df_wide.sort_values(by=["fips_code", "report_year"])

METRICS = [
    "diabetes_pct",
    "obesity_pct",
    "bphigh_pct",
    "lpa_pct",
    "smoking_pct",
    "checkup_pct",
    "chd_pct",
    "kidney_pct",
    "sleep_pct",
    "mhlth_pct"
]

# --- Lag Features ---
for metric in METRICS:
    df_wide[f"{metric}_lag1"] = df_wide.groupby("fips_code")[metric].shift(1)

# --- Year-over-Year Change (required) ---
for metric in ["diabetes_pct", "obesity_pct"]:
    df_wide[f"{metric}_yoy_change"] = (
        df_wide[metric] - df_wide[f"{metric}_lag1"]
    )

# --- Rolling averages (bonus) ---
for metric in METRICS:
    df_wide[f"{metric}_rolling3"] = (
        df_wide.groupby("fips_code")[metric]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# --- County ranking ---
df_wide["diabetes_rank"] = df_wide.groupby("report_year")["diabetes_pct"] \
    .rank(method="min", ascending=False)

# --- Target variable ---
df_wide["diabetes_next_year"] = df_wide.groupby("fips_code")["diabetes_pct"].shift(-1)

df_wide["top_quintile_threshold"] = df_wide.groupby("report_year")["diabetes_pct"] \
    .transform(lambda x: x.quantile(0.8))

df_wide["target_top_quintile_next_year"] = (
    df_wide["diabetes_next_year"] >= df_wide["top_quintile_threshold"]
).astype(int)

# --- Drop invalid rows ---
df_model = df_wide.dropna(subset=["diabetes_pct_lag1", "diabetes_next_year"])

print(f"Model-ready dataset: {df_model.shape}")

# -------------------------------------------------------
# --- STEP 5: Save output ---
# -------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)
OUTPUT_PATH = "data/processed/county_features.csv"

df_model.to_csv(OUTPUT_PATH, index=False)

print(f"Saved to {OUTPUT_PATH}")
df_model.head()