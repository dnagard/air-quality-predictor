# %%
import json
from pathlib import Path

import hopsworks
import pandas as pd
from pydantic_settings import BaseSettings
import util

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

# %%
def process_air_quality(df: pd.DataFrame, location: dict) -> None:
    """
    Process air quality dataframe in place.

    Transforms the dataframe to have columns: [id, date, pm25]
    """
    df.rename(columns={"median": "pm25"}, inplace=True)
    df["date"] = df["date"].dt.date
    df["pm25"] = df["pm25"].astype("float32")
    df.drop(df.columns.difference(["date", "pm25"]), axis=1, inplace=True)
    df.dropna(inplace=True)
    df["id"] = location["id"]

# %%
def load_air_quality_data(locations: dict) -> pd.DataFrame:
    """Load and process air quality data for all locations."""
    dfs = []

    for location_id in locations:
        file_path = Path(f"data/{location_id}.csv")
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found")

        print(f"Processing {location_id}")
        df = pd.read_csv(
            file_path, comment="#", skipinitialspace=True, parse_dates=["date"]
        )
        process_air_quality(df, locations[location_id])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


air_quality_df = load_air_quality_data(locations)

"""Lagged feature implementation"""

air_quality_df = air_quality_df.sort_values(["id", "date"])
grouped = air_quality_df.groupby("id", group_keys=False)
air_quality_df["pm25_lag_1"] = grouped["pm25"].shift(1)
air_quality_df["pm25_lag_2"] = grouped["pm25"].shift(2)
air_quality_df["pm25_lag_3"] = grouped["pm25"].shift(3)
# 3-day rolling mean (including current day)
air_quality_df["pm25_roll_3"] = (
    grouped["pm25"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)


"""Convert date column to datetime and extract day of week, latitude, longitude for more features to train on"""
air_quality_df["date"] = pd.to_datetime(air_quality_df["date"])

air_quality_df["day_of_week"] = air_quality_df["date"].dt.weekday

air_quality_df["latitude"] = air_quality_df["id"].map(
    lambda x: float(locations[x]["latitude"])
)
air_quality_df["longitude"] = air_quality_df["id"].map(
    lambda x: float(locations[x]["longitude"])
)

air_quality_df.info()

# %%
"""Upload to Hopsworks Feature Store. May need to change this to work with your setup.
I don't know which version you're on"""

project = hopsworks.login(engine="python", project="matcov")
fs = project.get_feature_store()

air_quality_fg = fs.get_or_create_feature_group(
    name="air_quality",
    description="Air Quality characteristics of each day",
    version=11,
    primary_key=["id"],
    event_time="date",
)
air_quality_fg
air_quality_df

# %%
air_quality_fg.insert(air_quality_df)

# %%

"""Update feature descriptions"""
air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
air_quality_fg.update_feature_description("pm25", "Particles less than 2.5 micrometers in diameter (fine particles) pose health risk")
air_quality_fg.update_feature_description("pm25_lag_1", "PM2.5 one day before the measurement")
air_quality_fg.update_feature_description("pm25_lag_2", "PM2.5 two days before the measurement")
air_quality_fg.update_feature_description("pm25_lag_3", "PM2.5 three days before the measurement")
air_quality_fg.update_feature_description("pm25_roll_3", "3-day rolling mean of PM2.5")


# %%
weather_df = util.get_historical(air_quality_df, locations)

"""Lagged feature implementation for weather data"""
weather_df = weather_df.sort_values(["id", "date"])
grouped = weather_df.groupby("id", group_keys=False)

weather_df["temperature_2m_mean_lag_1"] = grouped["temperature_2m_mean"].shift(1)
weather_df["precipitation_sum_lag_1"] = grouped["precipitation_sum"].shift(1)
weather_df["wind_speed_10m_max_lag_1"] = grouped["wind_speed_10m_max"].shift(1)
weather_df["wind_direction_10m_dominant_lag_1"] = grouped["wind_direction_10m_dominant"].shift(1)

"""Rolling averages for temperature and wind speed"""
weather_df["temp_roll_3"] = (
    grouped["temperature_2m_mean"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)
weather_df["wind_roll_3"] = (
    grouped["wind_speed_10m_max"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)
weather_df.info()

# %%

"""May need to update version to work with yours"""
weather_fg = fs.get_or_create_feature_group(
    name="weather",
    description="Weather characteristics of each day",
    version=11,
    primary_key=["id"],
    event_time="date",
)
weather_fg.insert(weather_df, wait=True)
# %%