
import os
from io import StringIO

import pandas as pd
import requests

try:
    import boto3
except ImportError:
    boto3 = None


S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = "processed/cdc_places"

DATASET_IDS = [
    "swc5-untb",  # 2025 release (contains 2022-2023 data)
    "fu4u-a9bh",  # 2024 release (contains 2021-2022 data)
    "h3ej-a9ec",  # 2023 release (contains 2020-2021 data)
    "duw2-7jbt",  # 2022 release (contains 2019-2020 data)
    "pqpp-u99h",  # 2021 release (contains 2018-2019 data)
    "dv4u-3x3q",  # 2020 release (contains 2017-2018 data)
]

BASE_URL = "https://data.cdc.gov/resource/{}.csv"

MEASURE_IDS = [
    "DIABETES",
    "OBESITY",
    "BPHIGH",
    "LPA",
    "CSMOKING",
    "CHECKUP",
    "CHD",
    "KIDNEY",
    "SLEEP",
    "MHLTH",
]

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
    "geolocation",
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
    "mhlth_pct",
]


def fetchDataset(datasetId: str) -> pd.DataFrame | None:
    try:
        measure_filter = " OR ".join([f"MeasureId='{m}'" for m in MEASURE_IDS])
        query = (
            f"?$where=({measure_filter}) AND DataValueTypeID='{VALUE_TYPE}'"
            "&$limit=50000&$offset=0"
        )
        url = BASE_URL.format(datasetId) + query

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), low_memory=False)

        if df.empty:
            print(f"[WARN] No data returned for dataset {datasetId}")
            return None

        df["source_dataset"] = datasetId
        print(f"[INFO] Loaded {datasetId}: {df.shape}")
        return df

    except Exception as exc:
        print(f"[ERROR] Failed to fetch {datasetId}: {exc}")
        return None


def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "locationid" not in df.columns and "latitude" in df.columns:
        df = df.rename(columns={"latitude": "locationid"})

    df = df[
        (df["measureid"].isin(MEASURE_IDS))
        & (df["datavaluetypeid"] == VALUE_TYPE)
    ]

    existing_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    df = df[existing_cols]

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["locationid"] = df["locationid"].astype(str)
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    df = df.dropna(subset=["year", "locationid", "measureid"])
    df["year"] = df["year"].astype(int)

    print(f"[STEP 2] Cleaned data: {df.shape}")
    return df


def pivotData(df: pd.DataFrame) -> pd.DataFrame:
    df_wide = (
        df.pivot_table(
            index=[
                "year",
                "stateabbr",
                "statedesc",
                "locationname",
                "locationid",
                "totalpopulation",
                "geolocation",
            ],
            columns="measureid",
            values="data_value",
        )
        .reset_index()
    )

    df_wide.columns.name = None

    df_wide = df_wide.rename(
        columns={
            "DIABETES": "diabetes_pct",
            "OBESITY": "obesity_pct",
            "BPHIGH": "bphigh_pct",
            "LPA": "lpa_pct",
            "CSMOKING": "smoking_pct",
            "CHECKUP": "checkup_pct",
            "CHD": "chd_pct",
            "KIDNEY": "kidney_pct",
            "SLEEP": "sleep_pct",
            "MHLTH": "mhlth_pct",
            "locationid": "fips_code",
            "year": "report_year",
        }
    )

    df_wide["county_year_id"] = (
        df_wide["fips_code"].astype(str) + "_" + df_wide["report_year"].astype(str)
    )

    before = len(df_wide)
    df_wide = df_wide.dropna(subset=["diabetes_pct"])
    after = len(df_wide)

    print(f"[STEP 3] Dropped {before - after} rows missing diabetes_pct")
    print(f"[STEP 3] Pivoted data: {df_wide.shape}")
    return df_wide


def engineerFeatures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["fips_code", "report_year"]).copy()

    for metric in METRICS:
        if metric in df.columns:
            df[f"{metric}_lag1"] = df.groupby("fips_code")[metric].shift(1)

    for metric in ["diabetes_pct", "obesity_pct"]:
        lag_col = f"{metric}_lag1"
        if metric in df.columns and lag_col in df.columns:
            df[f"{metric}_yoy_change"] = df[metric] - df[lag_col]

    for metric in METRICS:
        if metric in df.columns:
            df[f"{metric}_rolling3"] = (
                df.groupby("fips_code")[metric]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    if "diabetes_pct" in df.columns:
        df["diabetes_rank"] = (
            df.groupby("report_year")["diabetes_pct"]
            .rank(method="min", ascending=False)
        )

    df["diabetes_next_year"] = df.groupby("fips_code")["diabetes_pct"].shift(-1)
    df["top_quintile_threshold"] = (
        df.groupby("report_year")["diabetes_pct"]
        .transform(lambda x: x.quantile(0.8))
    )

    df["target_top_quintile_next_year"] = (
        df["diabetes_next_year"] >= df["top_quintile_threshold"]
    ).astype(int)

    df_model = df.dropna(subset=["diabetes_pct_lag1", "diabetes_next_year"]).copy()

    print(f"[STEP 4] Feature engineered data: {df_model.shape}")
    return df_model


def runPipeline() -> pd.DataFrame:
    all_dfs = []

    for dataset_id in DATASET_IDS:
        df = fetchDataset(dataset_id)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No data retrieved from any CDC dataset")

    combined_dfs = pd.concat(all_dfs, ignore_index=True)
    combined_dfs.columns = combined_dfs.columns.str.lower().str.strip()

    dataset_priority = {dataset_id: i for i, dataset_id in enumerate(DATASET_IDS)}
    combined_dfs["priority"] = combined_dfs["source_dataset"].map(dataset_priority)

    combined_dfs = (
        combined_dfs.sort_values("priority")
        .drop_duplicates(subset=["year", "locationid", "measureid"], keep="first")
        .drop(columns=["priority"])
    )

    print(f"[INFO] After dedup: {combined_dfs.shape}")
    print(f"[INFO] Years in combined data: {sorted(combined_dfs['year'].dropna().unique())}")

    df_clean = cleanData(combined_dfs)
    df_wide = pivotData(df_clean)
    df_engineered = engineerFeatures(df_wide)

    return df_engineered


def saveOutput(df: pd.DataFrame, local_path: str = "data/processed/county_facts.csv") -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"[INFO] Saved locally -> {local_path}")

    if S3_BUCKET:
        if boto3 is None:
            raise ImportError("boto3 is required for S3 uploads but is not installed")

        s3 = boto3.client("s3")
        key = f"{S3_PREFIX}/county_facts.csv"
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=csv_buffer.getvalue(),
        )
        print(f"[INFO] Uploaded to S3 -> s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    df_final = runPipeline()
    saveOutput(df_final)
