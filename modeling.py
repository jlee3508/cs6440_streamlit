import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

# Path to the cleaned pipeline output from DataIngestionMerger.py
DATA_PATH = "data/processed/county_facts.csv"

# Output paths
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features used for modeling
FEATURES = [
    "diabetes_pct_lag1",
    "obesity_pct",
    "lpa_pct",
    "smoking_pct",
    "chd_pct",
    "mhlth_pct",
    "diabetes_pct_yoy_change",
    "diabetes_pct_rolling3"
]

# Human readable names for dashboard display
FEATURE_NAME_MAP = {
    "diabetes_pct_rolling3":   "3-year diabetes trend",
    "diabetes_pct_lag1":       "Prior year diabetes rate",
    "diabetes_pct_yoy_change": "Year over year change",
    "obesity_pct":             "Obesity rate",
    "lpa_pct":                 "Physical inactivity",
    "smoking_pct":             "Smoking rate",
    "chd_pct":                 "Heart disease rate",
    "mhlth_pct":               "Poor mental health"
}


# -------------------------------------------------------
# AUTO-RUN INGESTION IF DATA FILE DOESN'T EXIST
# DataIngestionMerger.py pulls from CDC API and saves county_facts.csv
# If the file already exists this step is skipped
# Flow: CDC API -> DataIngestionMerger.py -> county_facts.csv -> model_pipeline.py
# -------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("county_facts.csv not found - running ingestion pipeline first...")
    print("This will pull data from the CDC API and may take a few minutes...")

    merger_path = "DataIngestionMerger.py"

    if os.path.exists(merger_path):
        exec(open(merger_path).read())
        dfFinal = runPipeline()
        saveOutput(dfFinal)
        print("Ingestion complete - county_facts.csv saved")
    else:
        raise FileNotFoundError(
            "county_facts.csv not found and DataIngestionMerger.py is not available. "
            "Please run DataIngestionMerger.py first to generate the data file."
        )
else:
    print(f"Found existing data file at {DATA_PATH} - skipping ingestion")


# -------------------------------------------------------
# STEP 1: Load data from pipeline output
# -------------------------------------------------------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Years available: {sorted(df['report_year'].unique())}")


# -------------------------------------------------------
# STEP 2: Temporal train/test split
# Train on earlier years, test on most recent year
# This simulates real world prediction - model never sees future data
# -------------------------------------------------------
print("\nSplitting data...")
train = df[df["report_year"] < 2022]
test  = df[df["report_year"] == 2022]

X_train = train[FEATURES]
y_train = train["target_top_quintile_next_year"]

X_test = test[FEATURES]
y_test = test["target_top_quintile_next_year"]

print(f"Training set: {X_train.shape[0]} rows (years 2019-2021)")
print(f"Test set: {X_test.shape[0]} rows (year 2022)")
print(f"Target balance: {y_train.mean():.1%} high risk in training set")


# -------------------------------------------------------
# STEP 3: Scale features
# Logistic regression requires scaled inputs
# Scaler is fit on training data only to avoid data leakage
# -------------------------------------------------------
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# -------------------------------------------------------
# STEP 4: Train baseline logistic regression model
# Chosen as primary model - interpretable and performs well
# for public health use case where explainability matters
# -------------------------------------------------------
print("\nTraining logistic regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")


# -------------------------------------------------------
# STEP 5: Train XGBoost for comparison
# More complex model - used to validate logistic regression results
# Both models achieved AUC of 0.98 so logistic regression was kept
# -------------------------------------------------------
print("\nTraining XGBoost for comparison...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print(f"AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

print("\nUsing Logistic Regression as primary model (same AUC 0.98, more interpretable)")


# -------------------------------------------------------
# STEP 6: Feature importance
# Shows which factors most influence the prediction
# Saved as a chart for use in presentation/dashboard
# -------------------------------------------------------
print("\nGenerating feature importance chart...")
feature_names = X_train.columns.tolist()
coefficients  = model.coef_[0]
sorted_idx    = np.argsort(np.abs(coefficients))[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    [feature_names[i] for i in sorted_idx],
    [coefficients[i] for i in sorted_idx]
)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance - Logistic Regression")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.show()
print(f"Saved feature importance chart")


# -------------------------------------------------------
# STEP 7: Generate predictions for all historical years (2020-2022)
# Applies the trained model to every county for every year we have data
# -------------------------------------------------------
print("\nGenerating predictions for all years (2020-2022)...")

def get_top_factors(row):
    """Get top 3 features driving the prediction for a county"""
    contributions = {}
    for feature, coef in zip(feature_names, coefficients):
        if feature in row:
            contributions[feature] = abs(row[feature] * coef)
    top3 = sorted(contributions, key=contributions.get, reverse=True)[:3]
    return ", ".join([FEATURE_NAME_MAP.get(f, f) for f in top3])

X_all        = df[FEATURES]
X_all_scaled = scaler.transform(X_all)

output_all = df.copy()
output_all["risk_probability"] = model.predict_proba(X_all_scaled)[:, 1]
output_all["predicted_label"]  = model.predict(X_all_scaled)
output_all["is_forecast"]      = False
output_all["top_factors"]      = df.apply(get_top_factors, axis=1)

output_all = output_all[[
    "fips_code", "locationname", "stateabbr",
    "report_year", "diabetes_pct",
    "risk_probability", "predicted_label",
    "is_forecast", "top_factors"
]].sort_values(["fips_code", "report_year"])

print(f"Historical predictions: {output_all.shape}")
print(output_all["report_year"].value_counts().sort_index())


# -------------------------------------------------------
# STEP 8: Generate future year forecasts (2023, 2024, 2025)
# Uses linear extrapolation based on each county's recent yoy trend
# Risk scores come from the trained model using 2022 feature values
# -------------------------------------------------------
print("\nGenerating forecasts for 2023, 2024, 2025...")

df = df.sort_values(by=["fips_code", "report_year"])
rolling_avg = df.groupby("fips_code")["diabetes_pct_yoy_change"].rolling(window=2, min_periods=1).mean()
df["avg_yoy_change"] = rolling_avg.reset_index(level=0, drop=True)

latest       = df[df["report_year"] == 2022].copy()
factors_2022 = latest.copy()
factors_2022["top_factors"] = factors_2022.apply(get_top_factors, axis=1)

forecasts = []
for _, row in latest.iterrows():

    # beginning data row
    current_row = row[FEATURES].values.copy()
    projected_diabetes = row["diabetes_pct"]

    for future_year in [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]:
        years_ahead = future_year - 2022

        # Project diabetes rate forward using recent year over year change
        projected_diabetes += (row["avg_yoy_change"] * (0.7 ** years_ahead))

        #print(current_row)

        # Use 2022 feature values to estimate risk for future years
        current_row[0] = projected_diabetes
        features_row = current_row.reshape(1, -1)
        features_scaled = scaler.transform(features_row)
        risk_prob      = model.predict_proba(features_scaled)[0][1]
        risk_label     = model.predict(features_scaled)[0]

        # Use 2022 top factors for forecast years since features dont change
        top_factors = factors_2022[factors_2022["fips_code"] == row["fips_code"]]["top_factors"].values
        top_factors = top_factors[0] if len(top_factors) > 0 else ""

        forecasts.append({
            "fips_code":       row["fips_code"],
            "locationname":    row["locationname"],
            "stateabbr":       row["stateabbr"],
            "report_year":     future_year,
            "diabetes_pct":    round(projected_diabetes, 1),
            "risk_probability": round(risk_prob, 4),
            "predicted_label": int(risk_label),
            "is_forecast":     True,
            "top_factors":     top_factors
        })

forecast_df = pd.DataFrame(forecasts)
print(f"Forecast predictions: {forecast_df.shape}")
print(forecast_df["report_year"].value_counts().sort_index())


# -------------------------------------------------------
# STEP 9: Combine historical + forecast into one file
# This is the main file the dashboard will load
# is_forecast column tells the dashboard which rows are real vs projected
# -------------------------------------------------------
print("\nCombining historical and forecast data...")
combined = pd.concat([output_all, forecast_df], ignore_index=True)
combined = combined.sort_values(["fips_code", "report_year"])

print(f"Combined dataset: {combined.shape}")
print(combined["report_year"].value_counts().sort_index())


# -------------------------------------------------------
# STEP 10: Build county summary table
# One clean row per county with everything the dashboard needs
# for the county detail view and risk table
# -------------------------------------------------------
print("\nBuilding county summary table...")

summary    = combined[combined["report_year"] == 2022].copy()
trend_data = df[df["report_year"] == 2022][["fips_code", "diabetes_pct_yoy_change"]].copy()
summary    = summary.merge(trend_data, on="fips_code", how="left")

def get_trend(yoy):
    """Classify year over year change as Increasing, Decreasing, or Stable"""
    if pd.isna(yoy):
        return "Unknown"
    elif yoy > 0.5:
        return "Increasing"
    elif yoy < -0.5:
        return "Decreasing"
    else:
        return "Stable"

summary["trend"] = summary["diabetes_pct_yoy_change"].apply(get_trend)

# Add 2025 projected rate for each county
projected_2025 = combined[combined["report_year"] == 2025][["fips_code", "diabetes_pct"]].rename(
    columns={"diabetes_pct": "projected_2025_pct"}
)
summary = summary.merge(projected_2025, on="fips_code", how="left")

county_summary = summary[[
    "fips_code",
    "locationname",
    "stateabbr",
    "diabetes_pct",
    "trend",
    "projected_2025_pct",
    "risk_probability",
    "predicted_label",
    "top_factors"
]].rename(columns={
    "diabetes_pct":    "current_diabetes_pct",
    "predicted_label": "high_risk"
}).sort_values("risk_probability", ascending=False)

print(f"County summary: {county_summary.shape}")
print(county_summary.head(5).to_string())


# -------------------------------------------------------
# SAVE ALL OUTPUTS
# 4 files handed off to the dashboard:
# 1. model_predictions_full.csv - all counties all years + forecasts
# 2. county_summary.csv - one row per county with full risk profile
# 3. logistic_model.pkl - saved trained model
# 4. scaler.pkl - saved feature scaler
# -------------------------------------------------------
print("\nSaving all outputs...")

combined.to_csv(f"{OUTPUT_DIR}/model_predictions_full.csv", index=False)
print(f"Saved model_predictions_full.csv: {combined.shape[0]} rows, years 2020-2025")

county_summary.to_csv(f"{OUTPUT_DIR}/county_summary.csv", index=False)
print(f"Saved county_summary.csv: {county_summary.shape[0]} counties")

joblib.dump(model,  f"{OUTPUT_DIR}/logistic_model.pkl")
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
print("Saved logistic_model.pkl and scaler.pkl")

print("\n--- SUMMARY ---")
print(f"model_predictions_full.csv : {combined.shape[0]} rows, years 2020-2030")
print(f"county_summary.csv         : {county_summary.shape[0]} counties with full risk profile")
print(f"logistic_model.pkl         : trained model, AUC 0.98")
print(f"scaler.pkl                 : feature scaler")
print(f"\nColumns in predictions : {combined.columns.tolist()}")
print(f"Columns in summary     : {county_summary.columns.tolist()}")