import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import DataIngestionMerger

# Load in the cleaned data
df = pd.read_csv('./data/processed/county_facts.csv')

#Drop missing data (will be fully cleaned with data ingestion already)
#df = df.fillna(df.median())
df.columns = df.columns.str.strip()
#print(df.columns.tolist())

# Get features
features = df[['bphigh_pct', 'chd_pct', 'checkup_pct', 'smoking_pct','lpa_pct','mhlth_pct','obesity_pct','sleep_pct']]
features = features.fillna(features.median())

# Get target
target = df['diabetes_pct']
target = target.fillna(target.median())

# Split into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1)



### Running with ingested data
ingested_df = df#DataIngestionMerger.runPipeline()

feature_list = ['diabetes_pct',
                 'diabetes_pct_yoy_change',
                 'obesity_pct_yoy_change',
                 'diabetes_pct_rolling3',
                 'obesity_pct_rolling3',
                 'lpa_pct_rolling3',
                 'smoking_pct_rolling3',
                 'checkup_pct_rolling3',
                 'chd_pct_rolling3',
                 'kidney_pct_rolling3',
                 'mhlth_pct_rolling3',
                 'diabetes_rank']

# Ingested features
ingested_features = ingested_df[feature_list]

# Get target features
ingested_targets = ingested_df[['diabetes_next_year', 'target_top_quintile_next_year']]

features_train, features_test, target_train, target_test = train_test_split(ingested_features, ingested_targets, test_size=0.2)


# Initialize and Train with max depth
model = RandomForestRegressor(n_estimators=100)
model.fit(features_train, target_train)

# Predict
predictions = model.predict(features_test)
print(f"Average Error: {mean_absolute_error(target_test, predictions)}")

importance = model.feature_importances_
summary = pd.Series(importance, index=[feature_list]).sort_values(ascending=False)
print(summary)