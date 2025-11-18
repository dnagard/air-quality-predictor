# %%
"""Batch inference pipeline for air quality predictions."""
import json
import os
from datetime import date

import hopsworks
import matplotlib.pyplot as plt
from pydantic_settings import BaseSettings, SettingsConfigDict
from xgboost import XGBRegressor
import pandas as pd

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()

# %%
def load_locations(filepath: str = "locations.json") -> dict:
    """Load location data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


locations = load_locations()

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()
mr = project.get_model_registry()

# %%
# 🔁 Use the latest trained model version here (adjust version if needed)
retrieved_model = mr.get_model(
    name="air_quality_xgboost_model",
    version=6,
)

print(f"✓ Retrieved model: {retrieved_model.name} v{retrieved_model.version}")

# Download model artifacts
saved_model_dir = retrieved_model.download()

# Load the XGBoost model
retrieved_xgboost_model = XGBRegressor()
retrieved_xgboost_model.load_model(os.path.join(saved_model_dir, "model.json"))
print("✓ Model loaded successfully")

# %%
today = date.today()
today_str = today.strftime("%Y-%m-%d")

# Use the same feature group versions as backfill/daily pipelines
weather_fg = fs.get_feature_group(name="weather", version=3)
air_quality_fg = fs.get_feature_group(name="air_quality", version=5)

# Get all forecast weather rows from today onward
batch_data = weather_fg.filter(weather_fg.date >= today_str).read()
print(f"✓ Retrieved {len(batch_data)} weather forecast records")

# Ensure datetime
batch_data["date"] = pd.to_datetime(batch_data["date"])

# %%
# 🔹 Get latest PM2.5 lag features and spatial info per station from air_quality FG
aq_hist = air_quality_fg.read()

aq_hist["date"] = pd.to_datetime(aq_hist["date"])
# Sort and keep the latest row per id
aq_hist = aq_hist.sort_values(["id", "date"])
latest_aq = aq_hist.groupby("id", as_index=False).tail(1)

# We only need these columns from air_quality
latest_aq = latest_aq[
    [
        "id",
        "pm25_lag_1",
        "pm25_lag_2",
        "pm25_lag_3",
        "pm25_roll_3",
        "latitude",
        "longitude",
    ]
]

# Merge latest PM2.5 lags + spatial features into the forecast weather rows
batch_data = batch_data.merge(latest_aq, on="id", how="left")

# %%
# 🔹 Add calendar features based on forecast date
batch_data["day_of_week"] = batch_data["date"].dt.weekday
batch_data["is_weekend"] = (batch_data["day_of_week"] >= 5).astype(int)

# %%
# 🔹 Rename weather columns to match training feature names
rename_map = {
    "temperature_2m_mean": "weather_temperature_2m_mean",
    "precipitation_sum": "weather_precipitation_sum",
    "wind_speed_10m_max": "weather_wind_speed_10m_max",
    "wind_direction_10m_dominant": "weather_wind_direction_10m_dominant",
    "temperature_2m_mean_lag_1": "weather_temperature_2m_mean_lag_1",
    "precipitation_sum_lag_1": "weather_precipitation_sum_lag_1",
    "wind_speed_10m_max_lag_1": "weather_wind_speed_10m_max_lag_1",
    "wind_direction_10m_dominant_lag_1": "weather_wind_direction_10m_dominant_lag_1",
    "temp_roll_3": "weather_temp_roll_3",
    "wind_roll_3": "weather_wind_roll_3",
}

batch_data = batch_data.rename(columns=rename_map)

# %%
# 🔹 Build feature matrix exactly like in training
feature_columns = [
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_roll_3",
    "day_of_week",
    "is_weekend",
    "latitude",
    "longitude",
    "weather_temperature_2m_mean",
    "weather_precipitation_sum",
    "weather_wind_speed_10m_max",
    "weather_wind_direction_10m_dominant",
    "weather_temperature_2m_mean_lag_1",
    "weather_precipitation_sum_lag_1",
    "weather_wind_speed_10m_max_lag_1",
    "weather_wind_direction_10m_dominant_lag_1",
    "weather_temp_roll_3",
    "weather_wind_roll_3",
]

X_batch = batch_data[feature_columns].copy()

print(f"Features for prediction: {X_batch.columns.tolist()}")
print(f"Batch shape: {X_batch.shape}")

# %%
# 🔹 Make predictions
predictions = retrieved_xgboost_model.predict(X_batch)

# Add predictions to batch data
batch_data["predicted_pm25"] = predictions
batch_data["forecast_date"] = today

print(f"✓ Generated {len(predictions)} predictions")
print(batch_data[["id", "date", "predicted_pm25"]].head())

# %%
# 🔹 Save predictions to feature group
forecast_data = batch_data[["id", "date", "predicted_pm25", "forecast_date"]]

forecasts_fg = fs.get_or_create_feature_group(
    name="air_quality_forecasts",
    description="Daily air quality predictions for monitoring",
    version=1,
    primary_key=["id", "forecast_date"],
    event_time="date",
)

forecasts_fg.insert(forecast_data, write_options={"wait_for_job": True})
print("✓ Predictions saved to Hopsworks")

# %%
# Plotting (unchanged except that we now use enriched `batch_data`)
images_dir = "model/images/forecasts"
os.makedirs(images_dir, exist_ok=True)

for location_id, location in locations.items():
    location_forecast = batch_data[batch_data["id"] == location_id].copy()

    if location_forecast.empty:
        print(f"⚠ No forecast data for {location['city']}")
        continue

    location_forecast = location_forecast.sort_values("date")

    plt.figure(figsize=(12, 6))
    plt.plot(
        location_forecast["date"],
        location_forecast["predicted_pm25"],
        marker="o",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel("Date")
    plt.ylabel("Predicted PM2.5 (μg/m³)")
    plt.title(f"Air Quality Forecast - {location['city']}, {location['country']}")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(images_dir, f"pm25_forecast_{location['city']}.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()

    print(f"✓ Saved forecast plot for {location['city']}")

# %%
# Optional: upload plots (unchanged)
try:
    dataset_api = project.get_dataset_api()
    today_str = today.strftime("%Y-%m-%d")

    if not dataset_api.exists("Resources/airquality"):
        dataset_api.mkdir("Resources/airquality")

    for location_id, location in locations.items():
        plot_path = os.path.join(images_dir, f"pm25_forecast_{location['city']}.png")

        if os.path.exists(plot_path):
            dataset_api.upload(
                plot_path,
                f"Resources/airquality/{location['city']}_{today_str}",
                overwrite=True,
            )

    proj_url = project.get_url()
    print("\n✓ Forecast plots uploaded to Hopsworks")
    print(f"View at: {proj_url}/settings/fb/path/Resources/airquality")

except Exception as e:
    print(f"⚠ Could not upload to Hopsworks dataset: {e}")

# %%
print(f"""
Batch Inference Summary:
-----------------------
Forecast date: {today}
Locations: {len(locations)}
Total predictions: {len(predictions)}
Predictions per location: ~{len(predictions) // len(locations)}
Saved to feature group: air_quality_forecasts v1
""")
# %%
