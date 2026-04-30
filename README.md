
# Diabetes Progression Risk Dashboard

CS6440 Group 34

Live App: https://diabetes-dashboard-group34.streamlit.app/

> Note on GitHub repositories: Streamlit Community Cloud cannot deploy from private repositories. The live app is connected to a separate public GitHub mirror. This GT repository is the authoritative source for all source code and documentation. The public repository that Streamlit pulls from can be found at: https://github.com/jlee3508/cs6440\_streamlit.

---

## Repository Layout

### Source Code

```

├── app.py           # Streamlit dashboard entry point

├── DataIngestionMerger\_connected.py  # Primary ingestion pipeline

├── DataIngestionMerger.py         # Standalone ingestion pipeline

├── DataIngestion.py              # Early prototype ingestion script

├── DataIngestionMergerNotebook.py   # Notebook-friendly ingestion version

├── modeling.py                     # Full modeling pipeline (LR + XGBoost + forecasting)

├── RandomForestModel.py            # Exploratory random forest model

├── requirements.txt                # Python dependencies

└── README.md                        

```


### Data


No large data files are committed to this repository. The app pulls live data from the CDC PLACES API at runtime.

---


CDC PLACES Dataset

- Source: CDC PLACES — Local Data for Better Health, County Data (https://chronicdata.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb)

- Coverage: County-level health estimates for all US counties, 2017–2023 data years across six annual releases

- License: Public domain (CDC open data)

- The app ingests six dataset IDs spanning 2020–2025 releases; dataset IDs are listed in `DataIngestionMerger\_connected.py`


Processed outputs (generated at runtime or by running `modeling.py` locally, not committed):


```

data/processed/

├── county_facts.csv              # Feature-engineered county-year dataset

├── model_predictions_full.csv    # Historical predictions + 2023–2030 forecasts

├── county_summary.csv            # One row per county with full risk profile

├── logistic_model.pkl            # Trained logistic regression model

└── scaler.pkl                    # Fitted StandardScaler

```


---


## Architecture


See [`docs/data_diagram.png`](docs/data_diagram.png) for the full data flow diagram showing the end-to-end transformation pipeline from CDC source data to dashboard output.(Sprint 2 deliverable 2.3).


---


## Deployment


### Live Environment


The app is deployed on Streamlit Community Cloud at:


https://diabetes-dashboard-group34.streamlit.app/


Streamlit Community Cloud requires a public GitHub repository. A public mirror is maintained for deployment purposes. This GT repository remains the authoritative source. The public repository that Streamlit pulls from can be found at: https://github.com/jlee3508/cs6440\_streamlit. 


### Running Locally


Prerequisites: Python 3.10+


```bash

# 1. Clone this repository

git clone <https://github.gatech.edu/thumke3/CS6440-group-project>

cd <repo-directory>

# 2. Install dependencies

pip install -r requirements.txt

# 3. Launch the dashboard

streamlit run app.py

```

On first launch without pre-generated data, the app will call the CDC API and build processed CSVs automatically. This takes a few minutes. Subsequent launches load from the cached files unless you click Refresh Data in the UI.

`requirements.txt` in this repository reflects the Streamlit Community Cloud deployment. Install locally with `pip install -r requirements.txt`.

---
