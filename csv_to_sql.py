import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Float, Integer, DateTime
import os

DB_USER = "postgres"
DB_PASSWORD = "oneplusz2"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "wind_app"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# # ==================== WIND DATA ====================
# csv_wind_path = "C:\\Users\\gobli\\Projects\\wind_app\\wind_app_fastapi\\wind_model\\wind_df.csv"

# df_wind = pd.read_csv(csv_wind_path, parse_dates=['created_at'])

# # Подстановка обязательного внешнего ключа (если в CSV его нет)
# if "wind_turbine_id" not in df_wind.columns:
#     df_wind["wind_turbine_id"] = 1

# dtype_wind = {
#     'created_at': DateTime(),
#     'wind_speed_ref': Float(),
#     'wind_speed': Float(),
#     'power_raw_W': Float(),
#     'power_curve_W': Float(),
#     'power_ac_W': Float(),
#     'rho': Float(),
#     'rotor_diameter_m': Float(),
#     'rated_power_W': Float(),
#     'wind_turbine_id': Integer()
# }

# print("Загрузка данных в таблицу wind_data...")
# df_wind.to_sql(
#     name='wind_data',
#     con=engine,
#     if_exists='append',
#     index=False,
#     dtype=dtype_wind,
#     chunksize=5000
# )
# print("Ветровые данные успешно загружены.")


# # ==================== SOLAR DATA ====================
# ==================== SOLAR DATA ====================
csv_solar_path = "C:\\Users\\gobli\\Projects\\wind_app\\wind_app_fastapi\\solar_model\\solar_df.csv"
df_solar = pd.read_csv(csv_solar_path, parse_dates=['created_at'])

# Обязательные поля solar_data
if "system_id" not in df_solar.columns:
    df_solar["system_id"] = 1

# ВАЖНО: выбираем ТОЛЬКО нужные столбцы (те, что есть в таблице)
columns_to_insert = [
    'created_at', 'ghi', 'dni', 'dhi', 'poa_global',
    'temp_air', 'wind_speed', 'temp_cell',
    'dc_power', 'ac_power', 'cloud_factor',
    'system_id'
]

# Если каких-то колонок нет в CSV — будет ошибка, но у вас они есть
df_solar = df_solar[columns_to_insert]

dtype_solar = {
    'created_at': DateTime(),
    'ghi': Float(),
    'dni': Float(),
    'dhi': Float(),
    'poa_global': Float(),
    'temp_air': Float(),
    'wind_speed': Float(),
    'temp_cell': Float(),
    'dc_power': Float(),
    'ac_power': Float(),
    'cloud_factor': Float(),
    'system_id': Integer()
}

print("Загрузка данных в таблицу solar_data...")
df_solar.to_sql(
    name='solar_data',
    con=engine,
    if_exists='append',
    index=False,
    dtype=dtype_solar,
    chunksize=5000
)
print("Солнечные данные успешно загружены.")