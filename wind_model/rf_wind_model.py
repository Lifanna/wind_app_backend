import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from wind_model.wind_model import WindModel


def create_lag_features(df, col, lags=24):
    """Добавляет лаги и агрегированные признаки за предыдущие часы."""
    for lag in range(1, lags + 1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # агрегированные статистики
    for window in [6, 12]:
        df[f"{col}_mean_{window}h"] = df[col].rolling(window).mean()
        df[f"{col}_std_{window}h"] = df[col].rolling(window).std()
        df[f"{col}_min_{window}h"] = df[col].rolling(window).min()
        df[f"{col}_max_{window}h"] = df[col].rolling(window).max()

    df = df.dropna()
    return df


def train_wind_forecast_model(df, forecast_horizon=6, save_path="wind_model.pkl"):
    """Обучает LightGBM и сохраняет модель в pickle."""
    df = create_lag_features(df, "wind_speed", lags=24)

    X = df.drop(columns=["wind_speed"])
    y = df["wind_speed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"✅ Обучение завершено:")
    print(f"R² = {r2_score(y_test, y_pred):.3f}, MAE = {mean_absolute_error(y_test, y_pred):.2f} m/s")

    # сохраняем модель
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Модель сохранена в {save_path}")

    # Прогноз на следующие 6 часов
    last_row = df.iloc[-24:].copy()
    forecast = []
    input_row = last_row.copy()

    for step in range(forecast_horizon):
        x_pred = input_row.drop(columns=["wind_speed"]).iloc[-1:]
        next_speed = model.predict(x_pred)[0]
        forecast.append(next_speed)

        # сдвигаем лаги
        new_row = input_row.iloc[-1:].copy()
        new_row["wind_speed"] = next_speed
        for lag in range(24, 1, -1):
            new_row[f"wind_speed_lag{lag}"] = input_row[f"wind_speed_lag{lag-1}"].iloc[-1]
        new_row["wind_speed_lag1"] = next_speed

        # обновляем агрегированные признаки
        history = pd.concat([input_row, new_row]).iloc[-12:]
        for window in [6, 12]:
            new_row[f"wind_speed_mean_{window}h"] = history["wind_speed"].tail(window).mean()
            new_row[f"wind_speed_std_{window}h"] = history["wind_speed"].tail(window).std()
            new_row[f"wind_speed_min_{window}h"] = history["wind_speed"].tail(window).min()
            new_row[f"wind_speed_max_{window}h"] = history["wind_speed"].tail(window).max()

        input_row = pd.concat([input_row, new_row]).iloc[1:]

    forecast_index = pd.date_range(
        df.index[-1] + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq="1H",
        tz="Asia/Almaty"
    )

    forecast_df = pd.DataFrame({"wind_speed_forecast": forecast}, index=forecast_index)
    return forecast_df


def load_and_forecast(df, forecast_horizon=6, model_path="wind_model.pkl"):
    """Загружает модель из pickle и делает прогноз."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"📦 Модель загружена из {model_path}")

    df = create_lag_features(df, "wind_speed", lags=24)
    last_row = df.iloc[-24:].copy()
    forecast = []
    input_row = last_row.copy()

    for step in range(forecast_horizon):
        x_pred = input_row.drop(columns=["wind_speed"]).iloc[-1:]
        next_speed = model.predict(x_pred)[0]
        forecast.append(next_speed)

        new_row = input_row.iloc[-1:].copy()
        new_row["wind_speed"] = next_speed
        for lag in range(24, 1, -1):
            new_row[f"wind_speed_lag{lag}"] = input_row[f"wind_speed_lag{lag-1}"].iloc[-1]
        new_row["wind_speed_lag1"] = next_speed

        input_row = pd.concat([input_row, new_row]).iloc[1:]

    forecast_index = pd.date_range(
        df.index[-1] + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq="1H",
        tz="Asia/Almaty"
    )

    forecast_df = pd.DataFrame({"wind_speed_forecast": forecast}, index=forecast_index)
    return forecast_df


def estimate_forecast_power(df, forecast_df, wind_model):
    """Переводит прогноз скорости ветра в прогноз мощности."""
    v_hub = forecast_df["wind_speed_forecast"].values
    power_curve = wind_model._power_curve(v_hub)
    forecast_df["power_ac_W_forecast"] = power_curve * wind_model.eta
    return forecast_df


if __name__ == "__main__":
    # 1️⃣ Генерация исторических данных
    wind_model = WindModel(
        mean_wind_annual=4.8,
        hub_height=50.0,
        rated_power_kw=50.0,
        cp=0.42,
        efficiency=0.95
    )

    df = wind_model.generate(start="2024-01-01", end="2024-12-31", freq="1h")
    df = df.drop(columns=["turbine_status"])
    df.to_csv("wind_df.csv")

    # 2️⃣ Обучение + сохранение модели
    forecast_df = train_wind_forecast_model(df, forecast_horizon=6)

    # 3️⃣ Загрузка и прогноз из pickle
    forecast_from_pickle = load_and_forecast(df, forecast_horizon=6)

    # 4️⃣ Переводим в мощность
    forecast_df = estimate_forecast_power(df, forecast_df, wind_model)
    forecast_from_pickle = estimate_forecast_power(df, forecast_from_pickle, wind_model)

    print("\nПрогноз из сохранённой модели:")
    print(forecast_from_pickle)

    # 5️⃣ Визуализация
    plt.figure(figsize=(10, 4))
    plt.plot(df.index[-48:], df["wind_speed"].iloc[-48:], label="История")
    plt.plot(forecast_df.index, forecast_df["wind_speed_forecast"], "r--", label="Прогноз (новая модель)")
    plt.plot(forecast_from_pickle.index, forecast_from_pickle["wind_speed_forecast"], "g--", label="Прогноз (pickle)")
    plt.title("Прогноз скорости ветра (6 часов вперёд)")
    plt.legend()
    plt.show()
