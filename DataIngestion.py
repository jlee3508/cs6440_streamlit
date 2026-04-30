import json
import os
import boto3
import requests
import pandas as pd
from io import StringIO
from datetime import datetime

# ==============================
# ENVIRONMENT VARIABLES
# ==============================
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = "raw/cdc_places"

# ==============================
# CDC DATASET IDS (ADD MORE HERE)
# ==============================
DATASET_IDS = [
    "swc5-untb",  # newest 2025
    "duw2-7jbt",  # 2022
    "pqpp-u99h",  # 2021
    "dv4u-3x3q",  # 2020
    "h3ej-a9ec",  # 2023
    "fu4u-a9bh",  # 2024
    # add more as discovered, working in 500 cities data would be awesome,
    # but standardization appears to break down between sets over time
    # the further we go the worse it may get. So the effort for data 
    # massaging goes up up up. But on the other hand more data = more accurate
    # I also have worries about rows with discrepancies between them like if the 2024 
    # and 2023 data sets have info that conflicts about the same town but for the
    # same year. In those cases I guess we take the information from the newer set?
    # That may be fixable in the processing step pretty easily below, by ordering
    # the dataset ids from newest to oldest and then not overwriting as we iterate
    # through
]

BASE_URL = "https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/{}.csv" # This needs to change

all_dataframes = []

for dataset_id in DATASET_IDS:
    url = BASE_URL.format(dataset_id)

    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df["source_dataset"] = dataset_id  # track origin

    all_dataframes.append(df)

# Combine everything to a single frame
combined_df = pd.concat(all_dataframes, ignore_index=True)