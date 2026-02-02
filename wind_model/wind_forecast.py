# wind_forecast.py
import pandas as pd
import pickle
from wind_model.wind_model import WindModel
from wind_model.rf_wind_model import create_lag_features  # если в отдельном файле, импортируй отсюда

def forecast_wind(current_time, model_path="wind_model/wind_model.pkl"):
    """
    Возвращает DataFrame с прогнозом ветровой генерации (в кВт) на 6 часов вперёд.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Wind модель загружена.")

    # создаём wind_model для перевода скорости в мощность
    wind_model = WindModel(
        mean_wind_annual=4.8,
        hub_height=50.0,
        rated_power_kw=50.0,
        cp=0.42,
        efficiency=0.95
    )

    # генерируем последние 2 дня для лагов
    df = wind_model.generate(
        start=(current_time - pd.Timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S"),
        end=current_time.strftime("%Y-%m-%d %H:%M:%S"),
        freq="1h"
    )
    df = df.drop(columns=["turbine_status"])
    df = create_lag_features(df, "wind_speed", lags=24)

    # берём последние 24 часа
    last_row = df.iloc[-24:].copy()
    forecast = []
    input_row = last_row.copy()

    for step in range(6):
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
        current_time + pd.Timedelta(hours=1),
        periods=6,
        freq="1H",
        tz="Asia/Almaty"
    )

    forecast_df = pd.DataFrame({"wind_speed_forecast": forecast}, index=forecast_index)
    forecast_df["wind_power_kw_forecast"] = (
        wind_model._power_curve(forecast_df["wind_speed_forecast"].values)
        * wind_model.eta / 1000  # в кВт
    )
    return forecast_df[["wind_speed_forecast", "wind_power_kw_forecast"]]

# пример самостоятельного запуска
if __name__ == "__main__":
    current_time = pd.Timestamp("2024-12-31 00:00:00", tz="Asia/Almaty")
    forecast = forecast_wind(current_time)
    print("\n🔮 Прогноз ветровой генерации:")
    print(forecast)
