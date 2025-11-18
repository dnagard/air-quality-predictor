# %%
"""Batch inference pipeline for air quality predictions."""
import json
import os
from datetime import date, datetime

import hopsworks
import matplotlib.pyplot as plt
import pandas as pd
from pydantic_settings import BaseSettings
from xgboost import XGBRegressor

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str
    aqicn_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()

# %%
def load_locations(filepath: str = "locations.json") -> dict:
    """Load location data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


locations = load_locations()
locations

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()
mr = project.get_model_registry()

# %%
# Retrieve the trained model
retrieved_model = mr.get_model(
    name="air_quality_xgboost_model",
    version=1,
)

print(f"✓ Retrieved model: {retrieved_model.name} v{retrieved_model.version}")

# %%
# Download model artifacts
saved_model_dir = retrieved_model.download()

# Load the XGBoost model
retrieved_xgboost_model = XGBRegressor()
retrieved_xgboost_model.load_model(os.path.join(saved_model_dir, "model.json"))
print("✓ Model loaded successfully")

# %%
# Get weather forecast data for future predictions
today_str = date.today().strftime("%Y-%m-%d")
weather_fg = fs.get_feature_group(name="weather", version=2)

batch_data = weather_fg.filter(weather_fg.date >= today_str).read()
print(f"✓ Retrieved {len(batch_data)} weather forecast records")
batch_data.head()

# %%
# Prepare features for prediction - add weather_ prefix to match training
feature_columns = [
    "temperature_2m_mean",
    "precipitation_sum", 
    "wind_speed_10m_max",
    "wind_direction_10m_dominant"
]

# Create a copy with prefixed column names
X_batch = batch_data[feature_columns].copy()
X_batch.columns = ["weather_" + col for col in X_batch.columns]

print(f"Features for prediction: {X_batch.columns.tolist()}")

# Make predictions
predictions = retrieved_xgboost_model.predict(X_batch)

# Add predictions to batch data
batch_data["predicted_pm25"] = predictions
batch_data["forecast_date"] = date.today()

print(f"✓ Generated {len(predictions)} predictions")
batch_data[["id", "date", "predicted_pm25"]].head()

# %%
# Save predictions to feature group
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
# Create forecast plots for each location
images_dir = "model/images/forecasts"
os.makedirs(images_dir, exist_ok=True)

for location_id, location in locations.items():
    location_forecast = batch_data[batch_data["id"] == location_id].copy()
    
    if location_forecast.empty:
        print(f"⚠ No forecast data for {location['city']}")
        continue
    
    location_forecast = location_forecast.sort_values("date")
    
    plt.figure(figsize=(12, 6))
    plt.plot(location_forecast["date"], location_forecast["predicted_pm25"], 
             marker="o", linewidth=2, markersize=8)
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
# Optional: Upload plots to Hopsworks dataset
try:
    dataset_api = project.get_dataset_api()
    today_str = date.today().strftime("%Y-%m-%d")
    
    if not dataset_api.exists("Resources/airquality"):
        dataset_api.mkdir("Resources/airquality")
    
    for location_id, location in locations.items():
        plot_path = os.path.join(images_dir, f"pm25_forecast_{location['city']}.png")
        
        if os.path.exists(plot_path):
            dataset_api.upload(
                plot_path, 
                f"Resources/airquality/{location['city']}_{today_str}",
                overwrite=True
            )
    
    proj_url = project.get_url()
    print(f"\n✓ Forecast plots uploaded to Hopsworks")
    print(f"View at: {proj_url}/settings/fb/path/Resources/airquality")
    
except Exception as e:
    print(f"⚠ Could not upload to Hopsworks dataset: {e}")

# %%
print(f"""
Batch Inference Summary:
-----------------------
Forecast date: {date.today()}
Locations: {len(locations)}
Total predictions: {len(predictions)}
Predictions per location: ~{len(predictions) // len(locations)}
Saved to feature group: air_quality_forecasts v1
""")

# %%