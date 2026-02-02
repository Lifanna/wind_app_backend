import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
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
df = model.generate(start="2024-01-01", end="2024-03-31", freq="1h")  # 3 месяца для примера

# === 2. Подготовка целевой переменной: прогноз на +6 часов ===
df["ac_power_future"] = df["ac_power"].shift(-6)  # на 6 часов вперёд
df.dropna(inplace=True)

# === 3. Лаги и признаки ===
for lag in range(1, 4):  # добавим лаги за 3 предыдущих часа
    df[f"ac_power_lag{lag}"] = df["ac_power"].shift(lag)
df.dropna(inplace=True)

feature_cols = [
    "ghi", "dhi", "dni",
    "temp_air", "wind_speed",
    "cloud_factor",
    "ac_power",
    "ac_power_lag1", "ac_power_lag2", "ac_power_lag3"
]
X = df[feature_cols]
y = df["ac_power_future"]

# === 4. Разделение на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === 5. Модель RandomForest ===
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
# from lightgbm import LGBMRegressor
# rf = LGBMRegressor(
#     n_estimators=600,
#     learning_rate=0.01,
#     num_leaves=31,
#     subsample=0.8,
#     colsample_bytree=0.8
# )
# rf.fit(X_train, y_train)

# === 6. Предсказание ===
y_pred = rf.predict(X_test)

# === 7. Оценка ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R²: {r2:.3f}, MAE: {mae:.1f} W")

# === 8. Визуализация ===
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:200], label="Real (ac_power +6h)", color="black")
plt.plot(y_pred[:200], label="Predicted", color="orange", alpha=0.8)
plt.title("6-hour solar power forecast")
plt.legend()
plt.tight_layout()
plt.show()

# === 9. Важность признаков ===
feat_importance = pd.Series(rf.feature_importances_, index=feature_cols)
feat_importance.sort_values().plot.barh(figsize=(6,4), title="Feature Importance")
plt.tight_layout()
plt.show()
