# %%
import json
import os
from datetime import datetime, timedelta

import hopsworks
import matplotlib.pyplot as plt
import pandas as pd
from pydantic_settings import BaseSettings
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

import optuna


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

# %%
"""Check versions here again"""
air_quality_fg = fs.get_feature_group(name="air_quality", version=11)
weather_fg = fs.get_feature_group(name="weather", version=11)

# %%
selected_features = air_quality_fg.select(["id", "pm25", "date", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_roll_3", "day_of_week", "latitude", "longitude"]).join(
    weather_fg.select_all(), on="id"
)

# %%
feature_view = fs.get_or_create_feature_view(
    name="air_quality_fv",
    description="Weather features with air quality as the target",
    version=11,
    labels=["pm25"],
    query=selected_features,
)

# %%
test_start = datetime.today() - timedelta(days=7)
X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_start=test_start
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
X_train.head()

# %%
# Drop non-numeric columns for training
columns_to_drop = ["date", "id"]

# Check if weather_date exists and add it to drop list
if "weather_date" in X_train.columns:
    columns_to_drop.append("weather_date")

X_train_features = X_train.drop(columns=columns_to_drop)
X_test_features = X_test.drop(columns=columns_to_drop)

# Drop NaNs from training set
train_df = pd.concat([X_train_features, y_train], axis=1).dropna()
X_train_features = train_df.drop(columns=["pm25"])
y_train = train_df[["pm25"]]

# Drop NaNs from test set
test_df = pd.concat([X_test_features, y_test], axis=1).dropna()
X_test_features = test_df.drop(columns=["pm25"])
y_test = test_df[["pm25"]]

print(f"After dropping NaNs - Training: {len(X_train_features)}, Test: {len(X_test_features)}")
print(f"Features used: {list(X_train_features.columns)}")
# %%

"""Here is where we do hyperparameter tuning with Optuna"""
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    model.fit(X_train_features, y_train.iloc[:, 0])

    preds = model.predict(X_test_features)
    mse = mean_squared_error(y_test.iloc[:, 0], preds)
    return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(f"  Value (MSE): {study.best_value:.4f}")
print(f"  Params: {study.best_params}")

best_params = study.best_params
best_params["random_state"] = 42
best_params["tree_method"] = "hist"
best_params["n_jobs"] = -1

xgb_regressor = XGBRegressor(**best_params)
xgb_regressor.fit(X_train_features, y_train.iloc[:, 0])
print("✓ Model training completed with tuned hyperparameters")

# %%
y_pred = xgb_regressor.predict(X_test_features)

mse = mean_squared_error(y_test.iloc[:, 0], y_pred)
r2 = r2_score(y_test.iloc[:, 0], y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# %%
# Prepare results dataframe
results_df = y_test.copy()
results_df["predicted_pm25"] = y_pred
results_df["date"] = X_test["date"]
results_df["id"] = X_test["id"]
results_df = results_df.sort_values(by=["date"])
results_df.head()

# %%
# Create model directory
model_dir = "model"
images_dir = os.path.join(model_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# %%
# Plot hindcast for each location
for location_id, location in locations.items():
    location_df = results_df[results_df["id"] == location_id]
    
    if location_df.empty:
        print(f"⚠ No test data for {location['city']}")
        continue
    
    plt.figure(figsize=(12, 6))
    plt.plot(location_df["date"], location_df["pm25"], label="Actual PM2.5", marker="o")
    plt.plot(location_df["date"], location_df["predicted_pm25"], label="Predicted PM2.5", marker="x")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.title(f"Air Quality Hindcast - {location['city']}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(images_dir, f"pm25_hindcast_{location['city']}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✓ Saved plot for {location['city']}")

# %%
# Plot feature importance
plt.figure(figsize=(10, 8))
plot_importance(xgb_regressor, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()

feature_importance_path = os.path.join(images_dir, "feature_importance.png")
plt.savefig(feature_importance_path)
plt.close()
print("✓ Saved feature importance plot")

# %%
# Save model
model_path = os.path.join(model_dir, "model.json")
xgb_regressor.save_model(model_path)
print(f"✓ Saved model to {model_path}")

# %%
# Register model in Hopsworks
mr = project.get_model_registry()

metrics = {
    "MSE": str(mse),
    "R²": str(r2),
}

aq_model = mr.python.create_model(
    name="air_quality_xgboost_model",
    metrics=metrics,
    feature_view=feature_view,
    description="Air Quality (PM2.5) predictor using XGBoost",
)

aq_model.save(model_dir)
print("✓ Model registered in Hopsworks")

# %%
print(f"""
Training Pipeline Summary:
-------------------------
Training samples: {len(X_train_features)}
Test samples: {len(X_test_features)}
MSE: {mse:.4f}
R²: {r2:.4f}
Model saved to: {model_dir}
""")
# %%