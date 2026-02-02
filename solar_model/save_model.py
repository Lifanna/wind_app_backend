import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from solar_model import SolarModel

# === 1. Генерация данных ===
model = SolarModel(
    latitude=51.6210400,
    longitude=73.1108200,
    timezone="Asia/Almaty",
    tilt=30,
    azimuth=180,
    target_kw=11.0,
    module_power_stc=330
)
df = model.generate(start="2024-01-01", end="2024-03-31", freq="1h")

# === 2. Целевая переменная: прогноз на +6 часов ===
df["ac_power_future"] = df["ac_power"].shift(-6)
df.dropna(inplace=True)

# === 3. Лаги ===
for lag in range(1, 4):
    df[f"ac_power_lag{lag}"] = df["ac_power"].shift(lag)
df.dropna(inplace=True)

# === 4. Признаки ===
feature_cols = [
    "ghi", "dhi", "dni",
    "temp_air", "wind_speed",
    "cloud_factor",
    "ac_power",
    "ac_power_lag1", "ac_power_lag2", "ac_power_lag3"
]
X = df[feature_cols]
y = df["ac_power_future"]

# === 5. Разделение ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 6. Обучение модели ===
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# === 7. Оценка ===
y_pred = rf.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.3f}, MAE: {mean_absolute_error(y_test, y_pred):.1f} W")

# === 8. Сохранение модели ===
with open("solar_forecast_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("✅ Модель сохранена в solar_forecast_model.pkl")

# === 10. Визуализация для контроля ===
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[-100:], label="Real (+6h)", color="black")
plt.plot(y_pred[-100:], label="Predicted (+6h)", color="orange", alpha=0.8)
plt.title("6-hour Solar Power Forecast")
plt.legend()
plt.tight_layout()
plt.show()
