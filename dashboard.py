# %%
"""Simple Streamlit dashboard for air quality predictions."""
import json
from datetime import date, timedelta

import hopsworks
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

# %%
class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    hopsworks_api_key: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )


settings = Settings()

# %%
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load air quality records and predictions from Hopsworks."""
    project = hopsworks.login(api_key_value=settings.hopsworks_api_key)
    fs = project.get_feature_store()
    
    air_quality_fg = fs.get_feature_group(name="air_quality", version=2)
    forecasts_fg = fs.get_feature_group(name="air_quality_forecasts", version=1)
    
    # Load last 21 days of historical data
    last_21_days = (date.today() - timedelta(days=21)).strftime("%Y-%m-%d")
    
    aq_df = air_quality_fg.filter(air_quality_fg.date >= last_21_days).read()
    aq_df = aq_df.sort_values(by="date")
    
    # Load forecasts (past and future)
    forecast_df = forecasts_fg.read()
    forecast_df = forecast_df.sort_values(by="date")
    
    return aq_df, forecast_df


def get_aqi_color(pm25_value):
    """Get color based on PM2.5 value."""
    if pm25_value <= 50:
        return "#00E400"
    elif pm25_value <= 100:
        return "#FFFF00"
    elif pm25_value <= 150:
        return "#FF7E00"
    elif pm25_value <= 200:
        return "#FF0000"
    elif pm25_value <= 300:
        return "#8F3F97"
    else:
        return "#7E0023"


def get_aqi_level(pm25_value):
    """Get AQI level text based on PM2.5 value."""
    if pm25_value <= 50:
        return "Good"
    elif pm25_value <= 100:
        return "Moderate"
    elif pm25_value <= 150:
        return "Unhealthy (Sensitive)"
    elif pm25_value <= 200:
        return "Unhealthy"
    elif pm25_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def create_map(locations: dict, df_air_quality: pd.DataFrame, df_forecast: pd.DataFrame):
    """Create an interactive map showing all locations with current air quality."""
    map_data = []
    
    for location_id, location in locations.items():
        # Get latest actual measurement
        location_data = df_air_quality[df_air_quality["id"] == location_id]
        
        if not location_data.empty:
            latest_pm25 = location_data.iloc[-1]["pm25"]
            latest_date = location_data.iloc[-1]["date"]
            source = "Actual"
        else:
            # Fallback to forecast if no actual data
            forecast_location = df_forecast[df_forecast["id"] == location_id]
            if not forecast_location.empty:
                latest_pm25 = forecast_location.iloc[-1]["predicted_pm25"]
                latest_date = forecast_location.iloc[-1]["date"]
                source = "Forecast"
            else:
                continue
        
        map_data.append({
            "city": location["city"],
            "country": location["country"],
            "lat": float(location["latitude"]),
            "lon": float(location["longitude"]),
            "pm25": latest_pm25,
            "date": latest_date,
            "aqi_level": get_aqi_level(latest_pm25),
            "color": get_aqi_color(latest_pm25),
            "source": source,
            "location_id": location_id
        })
    
    df_map = pd.DataFrame(map_data)
    
    if df_map.empty:
        st.warning("No location data available for map")
        return
    
    # Create map with plotly
    fig = go.Figure()
    
    for _, row in df_map.iterrows():
        fig.add_trace(go.Scattermapbox(
            lat=[row["lat"]],
            lon=[row["lon"]],
            mode='markers+text',
            marker=dict(
                size=25,
                color=row["color"],
                opacity=0.9,
                sizemode='diameter'
            ),
            text=row["city"],
            textposition="top center",
            textfont=dict(size=10, color="black", family="Arial Black"),
            hovertemplate=(
                f"<b>{row['city']}</b><br>" +
                f"PM2.5: {row['pm25']:.1f} μg/m³<br>" +
                f"Level: {row['aqi_level']}<br>" +
                f"Date: {row['date']}<br>" +
                "<extra></extra>"
            ),
            name=row["city"],
            showlegend=False
        ))
    
    # Calculate center of all points
    center_lat = df_map["lat"].mean()
    center_lon = df_map["lon"].mean()
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=7
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        hovermode='closest',
        dragmode='pan'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_plot(location: dict, df_air_quality: pd.DataFrame, df_forecast: pd.DataFrame) -> None:
    """Create and display air quality visualization."""
    location_id = location["id"]
    
    # Filter data for this location
    hist_data = df_air_quality[df_air_quality["id"] == location_id].copy()
    forecast_data = df_forecast[df_forecast["id"] == location_id].copy()
    
    if hist_data.empty and forecast_data.empty:
        st.warning(f"No data available for {location['city']}")
        return
    
    # Convert all dates to datetime for consistency
    if not hist_data.empty:
        hist_data["date"] = pd.to_datetime(hist_data["date"])
    if not forecast_data.empty:
        forecast_data["date"] = pd.to_datetime(forecast_data["date"])
    
    fig = go.Figure()
    
    # Add historical actual data
    if not hist_data.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_data["date"],
                y=hist_data["pm25"],
                mode="lines+markers",
                name="Actual PM2.5",
                line=dict(color="#06A77D", width=2),
                marker=dict(size=6)
            )
        )
    
    # Add forecast data
    if not forecast_data.empty:
        # Separate past forecasts (hindcast) from future forecasts
        today = pd.Timestamp(date.today())
        past_forecasts = forecast_data[forecast_data["date"] < today]
        future_forecasts = forecast_data[forecast_data["date"] >= today]
        
        # Show past predictions (hindcast)
        if not past_forecasts.empty:
            fig.add_trace(
                go.Scatter(
                    x=past_forecasts["date"],
                    y=past_forecasts["predicted_pm25"],
                    mode="lines+markers",
                    name="Past Predictions",
                    line=dict(color="#D84797", width=2, dash="dash"),
                    marker=dict(size=6, symbol="x")
                )
            )
        
        # Show future predictions
        if not future_forecasts.empty:
            fig.add_trace(
                go.Scatter(
                    x=future_forecasts["date"],
                    y=future_forecasts["predicted_pm25"],
                    mode="lines+markers",
                    name="Future Forecast",
                    line=dict(color="#2E86AB", width=2, dash="dot"),
                    marker=dict(size=8)
                )
            )
    
    # Add vertical line for today using pd.Timestamp
    today_timestamp = pd.Timestamp(date.today())
    fig.add_vline(
        x=today_timestamp.timestamp() * 1000,  # Convert to milliseconds
        line_dash="solid",
        line_color="gray",
        line_width=2,
        annotation_text="Today",
        annotation_position="top",
        annotation=dict(
            font_size=12,
            font_color="gray"
        )
    )
    
    # Add AQI health guidelines as reference lines
    fig.add_hline(y=50, line_dash="dash", line_color="#00E400", line_width=1.5, opacity=0.6)
    fig.add_hline(y=100, line_dash="dash", line_color="#FFFF00", line_width=1.5, opacity=0.6)
    fig.add_hline(y=150, line_dash="dash", line_color="#FF7E00", line_width=1.5, opacity=0.6)
    fig.add_hline(y=200, line_dash="dash", line_color="#FF0000", line_width=1.5, opacity=0.6)
    fig.add_hline(y=300, line_dash="dash", line_color="#8F3F97", line_width=1.5, opacity=0.6)
    
    fig.update_layout(
        title=f"Air Quality in {location['city']}, {location['country']}",
        xaxis_title="Date",
        yaxis_title="PM2.5 (μg/m³)",
        hovermode="x unified",
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


# %%
# Main dashboard
st.set_page_config(page_title="Air Quality Dashboard", page_icon="🌍", layout="wide")

st.title("🌍 Air Quality Prediction Dashboard")

# Project introduction
st.markdown("""
This dashboard presents a machine learning system for predicting PM2.5 air quality levels across Swedish cities. 
Using historical weather and air quality data, an XGBoost model forecasts pollution levels up to 10 days in advance. 
The system runs daily automated pipelines to collect fresh data, generate predictions, and monitor model performance 
through hindcast analysis comparing predictions against actual measurements.
""")

# Load locations
with open("locations.json") as f:
    locations = json.load(f)

# Load data
with st.spinner("Loading data from Hopsworks..."):
    historical_data, forecast_data = load_data()

st.success("✓ Data loaded successfully!")

st.markdown("---")

# Map Section
st.markdown("### 🗺️ Air Quality Overview Map")
create_map(locations, historical_data, forecast_data)

st.markdown("---")

# Detailed plots section
st.markdown("### 📈 Detailed Forecasts by Location")

# Create plots for each location
for location_id, location in locations.items():
    create_plot(location, historical_data, forecast_data)
    st.markdown("---")

# AQI Guidelines Table
st.markdown("### 📊 Air Quality Index (AQI) Guidelines")

# Create styled HTML table
aqi_table_html = """
<style>
    .aqi-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
        box-shadow: 0 2px 3px rgba(0,0,0,0.1);
    }
    .aqi-table th {
        background-color: #f0f0f0;
        color: #333;
        text-align: left;
        padding: 12px;
        font-weight: bold;
        border: 1px solid #ddd;
    }
    .aqi-table td {
        padding: 12px;
        border: 1px solid #ddd;
    }
    .aqi-good {
        background-color: #00E400;
        color: white;
        font-weight: bold;
    }
    .aqi-moderate {
        background-color: #FFFF00;
        color: black;
        font-weight: bold;
    }
    .aqi-sensitive {
        background-color: #FF7E00;
        color: white;
        font-weight: bold;
    }
    .aqi-unhealthy {
        background-color: #FF0000;
        color: white;
        font-weight: bold;
    }
    .aqi-very-unhealthy {
        background-color: #8F3F97;
        color: white;
        font-weight: bold;
    }
    .aqi-hazardous {
        background-color: #7E0023;
        color: white;
        font-weight: bold;
    }
</style>

<table class="aqi-table">
    <thead>
        <tr>
            <th>AQI</th>
            <th>Air Pollution Level</th>
            <th>Health Implications</th>
            <th>Cautionary Statement (for PM2.5)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="aqi-good">0 - 50</td>
            <td class="aqi-good">Good</td>
            <td>Air quality is considered satisfactory, and air pollution poses little or no risk</td>
            <td>None</td>
        </tr>
        <tr>
            <td class="aqi-moderate">51 - 100</td>
            <td class="aqi-moderate">Moderate</td>
            <td>Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution.</td>
            <td>Active children and adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion.</td>
        </tr>
        <tr>
            <td class="aqi-sensitive">101 - 150</td>
            <td class="aqi-sensitive">Unhealthy for Sensitive Groups</td>
            <td>Members of sensitive groups may experience health effects. The general public is not likely to be affected.</td>
            <td>Active children and adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion.</td>
        </tr>
        <tr>
            <td class="aqi-unhealthy">151 - 200</td>
            <td class="aqi-unhealthy">Unhealthy</td>
            <td>Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects</td>
            <td>Active children and adults, and people with respiratory disease, such as asthma, should avoid prolonged outdoor exertion; everyone else, especially children, should limit prolonged outdoor exertion</td>
        </tr>
        <tr>
            <td class="aqi-very-unhealthy">201 - 300</td>
            <td class="aqi-very-unhealthy">Very Unhealthy</td>
            <td>Health warnings of emergency conditions. The entire population is more likely to be affected.</td>
            <td>Active children and adults, and people with respiratory disease, such as asthma, should avoid all outdoor exertion; everyone else, especially children, should limit outdoor exertion.</td>
        </tr>
        <tr>
            <td class="aqi-hazardous">300+</td>
            <td class="aqi-hazardous">Hazardous</td>
            <td>Health alert: everyone may experience more serious health effects</td>
            <td>Everyone should avoid all outdoor exertion</td>
        </tr>
    </tbody>
</table>
"""

st.markdown(aqi_table_html, unsafe_allow_html=True)

# %%