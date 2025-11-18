# %%
import json
import hopsworks
import pandas as pd
from datetime import date
from pydantic_settings import BaseSettings
import util
from pandas import to_datetime

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
air_quality_df = pd.DataFrame()
today = date.today()

for location_id, location in locations.items():
    try:
        aq_data = util.get_pm25(location_id, location, today, settings.aqicn_api_key)
        air_quality_df = pd.concat([air_quality_df, aq_data], ignore_index=True)
        print(f"✓ Fetched air quality for {location['city']}")
    except Exception as e:
        print(f"✗ Error for {location_id}: {e}")
        continue

air_quality_df.info()

# %%
weather_df = util.get_forecast(forecast_days=7, places=locations)
weather_df.info()

# %%
project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
fs = project.get_feature_store()

# %%
air_quality_fg = fs.get_feature_group(name="air_quality", version=5)
weather_fg = fs.get_feature_group(name="weather", version=3)

#%%

air_quality_df["date"] = to_datetime(air_quality_df["date"])

# Read historical air quality from FG
hist_aq = air_quality_fg.read()[["id", "date", "pm25"]]
hist_aq["date"] = to_datetime(hist_aq["date"])

# Combine historical + today
combined_aq = pd.concat([hist_aq, air_quality_df], ignore_index=True)
combined_aq = combined_aq.sort_values(["id", "date"])

# Group by sensor and compute lags/rolling
grouped_aq = combined_aq.groupby("id", group_keys=False)
combined_aq["pm25_lag_1"] = grouped_aq["pm25"].shift(1)
combined_aq["pm25_lag_2"] = grouped_aq["pm25"].shift(2)
combined_aq["pm25_lag_3"] = grouped_aq["pm25"].shift(3)
combined_aq["pm25_roll_3"] = (
    grouped_aq["pm25"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Keep only today's rows
today_ts = to_datetime(today)
new_aq = combined_aq[combined_aq["date"] == today_ts].copy()

# Calendar features
new_aq["day_of_week"] = new_aq["date"].dt.weekday
new_aq["is_weekend"] = (new_aq["day_of_week"] >= 5).astype(int)

# Spatial features
new_aq["latitude"] = new_aq["id"].map(lambda x: float(locations[x]["latitude"]))
new_aq["longitude"] = new_aq["id"].map(lambda x: float(locations[x]["longitude"]))

print("Air quality rows to insert today:")
new_aq.info()

#%%

# Ensure date is datetime for forecast data
weather_df["date"] = to_datetime(weather_df["date"])

# Read historical weather from FG
hist_weather = weather_fg.read()
hist_weather["date"] = to_datetime(hist_weather["date"])

# Keep only base weather columns (we recompute lags & rolls)
base_cols = [
    "id",
    "date",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
hist_weather = hist_weather[base_cols]

# Combine historical + new forecast
combined_weather = pd.concat([hist_weather, weather_df], ignore_index=True)
combined_weather = combined_weather.sort_values(["id", "date"])

# Group and compute weather lags / rolling means
grouped_w = combined_weather.groupby("id", group_keys=False)

combined_weather["temperature_2m_mean_lag_1"] = grouped_w["temperature_2m_mean"].shift(1)
combined_weather["precipitation_sum_lag_1"] = grouped_w["precipitation_sum"].shift(1)
combined_weather["wind_speed_10m_max_lag_1"] = grouped_w["wind_speed_10m_max"].shift(1)
combined_weather["wind_direction_10m_dominant_lag_1"] = grouped_w["wind_direction_10m_dominant"].shift(1)

combined_weather["temp_roll_3"] = (
    grouped_w["temperature_2m_mean"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)
combined_weather["wind_roll_3"] = (
    grouped_w["wind_speed_10m_max"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Keep only the forecast period (dates from the forecast_df)
forecast_dates = weather_df["date"].unique()
new_weather = combined_weather[combined_weather["date"].isin(forecast_dates)].copy()

print("Weather rows to insert (forecast):")
new_weather.info()

# %%
print(f"Inserting {len(new_aq)} air quality records...")
air_quality_fg.insert(new_aq)

# %%
print(f"Inserting {len(new_weather)} weather records...")
weather_fg.insert(new_weather, wait=True)
print("✓ Daily feature pipeline completed successfully!")

# %%