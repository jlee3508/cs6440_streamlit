import os
import boto3
import requests
import pandas as pd
from io import StringIO


# AWS (optional - leave None for local testing)
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = "processed/cdc_places"

# CDC dataset IDs ordered newest to oldest
# This order matters for deduplication — newer data takes priority
DATASET_IDS = [
    "swc5-untb",  # 2025 release (contains 2022-2023 data)
    "fu4u-a9bh",  # 2024 release (contains 2021-2022 data)
    "h3ej-a9ec",  # 2023 release (contains 2020-2021 data)
    "duw2-7jbt",  # 2022 release (contains 2019-2020 data)
    "pqpp-u99h",  # 2021 release (contains 2018-2019 data)
    "dv4u-3x3q",  # 2020 release (contains 2017-2018 data)
]

BASE_URL = "https://data.cdc.gov/resource/{}.csv"

# Measures to keep
# NOTE: these are uppercase values inside the measureid column
MEASURE_IDS = [
    "DIABETES", "OBESITY", "BPHIGH", "LPA",
    "CSMOKING", "CHECKUP", "CHD", "KIDNEY",
    "SLEEP", "MHLTH"
]

# NOTE: column names from API are lowercase, but values inside columns keep original casing
VALUE_TYPE = "CrdPrv"

COLUMNS_TO_KEEP = [
    "year",
    "stateabbr",
    "statedesc",
    "locationname",
    "locationid",
    "measureid",
    "data_value",
    "low_confidence_limit",
    "high_confidence_limit",
    "totalpopulation",
    "geolocation"
]

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


# -------------------------------------------------------
# STEP 1: Fetch each dataset from CDC API
# Filters at API level to only pull the measures we need
# -------------------------------------------------------
def fetchDataset(datasetId):
    try:
        measure_filter = " OR ".join([f"MeasureId='{m}'" for m in MEASURE_IDS])
        query = f"?$where=({measure_filter}) AND DataValueTypeID='{VALUE_TYPE}'&$limit=50000&$offset=0"

        url = BASE_URL.format(datasetId) + query

        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), low_memory=False)

        if df.empty:
            print(f"[WARN] No data returned for dataset {datasetId}")
            return None

        df["source_dataset"] = datasetId
        print(f"[INFO] Loaded {datasetId}: {df.shape}")

        return df

    except Exception as e:
        print(f"[ERROR] Failed to fetch {datasetId}: {e}")
        return None


# -------------------------------------------------------
# STEP 2: Clean and filter
# Normalizes column names, fixes dataset inconsistencies,
# filters to only the rows and columns we need
# -------------------------------------------------------
def cleanData(df):
    # Normalize all column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Fix 2020 dataset — uses 'latitude' instead of 'locationid'
    if "locationid" not in df.columns and "latitude" in df.columns:
        df = df.rename(columns={"latitude": "locationid"})

 
    df = df[
        (df["measureid"].isin(MEASURE_IDS)) &
        (df["datavaluetypeid"] == VALUE_TYPE)
    ]

    existingCols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    df = df[existingCols]

    df["year"] = df["year"].astype(int)
    df["locationid"] = df["locationid"].astype(str)
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    print(f"[STEP 2] Cleaned data: {df.shape}")
    return df


# -------------------------------------------------------
# STEP 3: Pivot to wide format
# One row per county per year, one column per measure
# Drops rows where diabetes is missing since its our core variable
# -------------------------------------------------------
def pivotData(df):
    dfWide = df.pivot_table(
        index=[
            "year", "stateabbr", "statedesc",
            "locationname", "locationid",
            "totalpopulation", "geolocation"
        ],
        columns="measureid",
        values="data_value"
    ).reset_index()

    dfWide.columns.name = None

    dfWide = dfWide.rename(columns={
        "DIABETES": "diabetes_pct",
        "OBESITY":  "obesity_pct",
        "BPHIGH":   "bphigh_pct",
        "LPA":      "lpa_pct",
        "CSMOKING": "smoking_pct",
        "CHECKUP":  "checkup_pct",
        "CHD":      "chd_pct",
        "KIDNEY":   "kidney_pct",
        "SLEEP":    "sleep_pct",
        "MHLTH":    "mhlth_pct",
        "locationid": "fips_code",
        "year":       "report_year"
    })

    # Composite primary key
    dfWide["county_year_id"] = (
        dfWide["fips_code"].astype(str) + "_" +
        dfWide["report_year"].astype(str)
    )

    # Drop rows where diabetes is missing — core variable, useless without it
    before = len(dfWide)
    dfWide = dfWide.dropna(subset=["diabetes_pct"])
    after = len(dfWide)
    print(f"[STEP 3] Dropped {before - after} rows missing diabetes_pct")
    print(f"[STEP 3] Pivoted data: {dfWide.shape}")

    return dfWide


# -------------------------------------------------------
# STEP 4: Feature engineering
# Creates lag variables, year-over-year change,
# rolling averages, county ranking, and target variable
# -------------------------------------------------------
def engineerFeatures(df):
    df = df.sort_values(by=["fips_code", "report_year"])

    # Lag features — prior year value for each metric
    for metric in METRICS:
        if metric in df.columns:
            df[f"{metric}_lag1"] = df.groupby("fips_code")[metric].shift(1)

    # Year-over-year change for diabetes and obesity
    for metric in ["diabetes_pct", "obesity_pct"]:
        if metric in df.columns and f"{metric}_lag1" in df.columns:
            df[f"{metric}_yoy_change"] = df[metric] - df[f"{metric}_lag1"]

    # Rolling 3-year average for each metric
    for metric in METRICS:
        if metric in df.columns:
            df[f"{metric}_rolling3"] = (
                df.groupby("fips_code")[metric]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # County diabetes ranking per year (1 = highest prevalence)
    if "diabetes_pct" in df.columns:
        df["diabetes_rank"] = df.groupby("report_year")["diabetes_pct"] \
            .rank(method="min", ascending=False)

    df["diabetes_next_year"] = df.groupby("fips_code")["diabetes_pct"].shift(-1)

    df["top_quintile_threshold"] = df.groupby("report_year")["diabetes_pct"] \
        .transform(lambda x: x.quantile(0.8))

    df["target_top_quintile_next_year"] = (
        df["diabetes_next_year"] >= df["top_quintile_threshold"]
    ).astype(int)

    df_model = df.dropna(subset=["diabetes_pct_lag1", "diabetes_next_year"])

    print(f"[STEP 4] Feature engineered data: {df_model.shape}")
    return df_model


# -------------------------------------------------------
# Run full pipeline
# -------------------------------------------------------
def runPipeline():
    allDfs = []

    # Step 1: Fetch all datasets
    for datasetId in DATASET_IDS:
        df = fetchDataset(datasetId)
        if df is not None:
            allDfs.append(df)

    if not allDfs:
        raise Exception("No data retrieved from any dataset")
    combinedDfs = pd.concat(allDfs, ignore_index=True)
    combinedDfs.columns = combinedDfs.columns.str.lower().str.strip()

    dataset_priority = {id: i for i, id in enumerate(DATASET_IDS)}
    combinedDfs["priority"] = combinedDfs["source_dataset"].map(dataset_priority)

    combinedDfs = combinedDfs.sort_values("priority").drop_duplicates(
        subset=["year", "locationid", "measureid"],
        keep="first"
    ).drop(columns=["priority"])

    print(f"After dedup: {combinedDfs.shape}")
    print(f"Years in combined data: {sorted(combinedDfs['year'].unique())}")

    # Step 2: Clean
    dfClean = cleanData(combinedDfs)

    # Step 3: Pivot
    dfWide = pivotData(dfClean)

    # Step 4: Engineer features
    dfEngineered = engineerFeatures(dfWide)

    return dfEngineered


# -------------------------------------------------------
# Save output locally and optionally to S3
# -------------------------------------------------------
def saveOutput(df):
    os.makedirs("data/processed", exist_ok=True)
    localPath = "data/processed/county_facts.csv"
    df.to_csv(localPath, index=False)
    print(f"[INFO] Saved locally -> {localPath}")

    # Upload to S3 if bucket is configured
    if S3_BUCKET:
        s3 = boto3.client("s3")
        key = f"{S3_PREFIX}/county_facts.csv"
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print(f"[INFO] Uploaded to S3 -> s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    dfFinal = runPipeline()
    saveOutput(dfFinal)