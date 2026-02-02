# solar_forecast.py
import pickle
import pandas as pd
from solar_model.solar_model import SolarModel

def forecast_solar(current_time, model_path="solar_model/solar_forecast_model.pkl"):
    """
    Возвращает DataFrame с прогнозом солнечной генерации на 6 часов вперёд.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Solar модель загружена.")

    # создаём SolarModel и генерируем данные (2 дня, чтобы были лаги)
    solar_gen = SolarModel(
        latitude=51.6210400,
        longitude=73.1108200,
        timezone="Asia/Almaty",
        tilt=30,
        azimuth=180,
        target_kw=11.0,
        module_power_stc=330
    )

    start = (current_time - pd.Timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")
    end = (current_time + pd.Timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")

    df_new = solar_gen.generate(start=start, end=end, freq="1h")
    df_new = df_new.copy()

    # признаки как при обучении
    for lag in range(1, 4):
        df_new[f"ac_power_lag{lag}"] = df_new["ac_power"].shift(lag)
    df_new.dropna(inplace=True)

    feature_cols = [
        "ghi", "dhi", "dni",
        "temp_air", "wind_speed",
        "cloud_factor",
        "ac_power",
        "ac_power_lag1", "ac_power_lag2", "ac_power_lag3"
    ]
    X_new = df_new[feature_cols]

    # прогноз
    df_new["forecast_ac_power_+6h"] = model.predict(X_new)
    df_new.loc[df_new["ghi"] == 0, "forecast_ac_power_+6h"] = 0.0

    # берём только 6 последних часов прогноза
    forecast_df = df_new[["ac_power", "forecast_ac_power_+6h"]].tail(6)
    forecast_df.rename(columns={"forecast_ac_power_+6h": "solar_power_forecast"}, inplace=True)

    print(forecast_df)

    return forecast_df

# пример самостоятельного запуска
if __name__ == "__main__":
    current_time = pd.Timestamp("2024-12-31 00:00:00", tz="Asia/Almaty")
    forecast = forecast_solar(current_time)
    print("\n🔮 Прогноз солнечной генерации:")
    print(forecast)
