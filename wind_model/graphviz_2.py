import matplotlib.pyplot as plt
import pandas as pd

# Загружаем CSV
df = pd.read_csv("wind_df.csv")

# Преобразуем в datetime с приведением к UTC
df['timestamp'] = pd.to_datetime(df['Unnamed: 0'], utc=True, errors='coerce')
df = df.set_index('timestamp').drop(columns=['Unnamed: 0'])

# Проверим, что теперь индекс правильный
print(df.index.dtype)

# Группируем по часу (по UTC)
hourly = df.groupby(df.index.hour)['power_ac_W'].mean()

# Визуализация
plt.figure(figsize=(10, 5))
plt.plot(hourly.index, hourly.values / 1000, color="#117A65", lw=2)
plt.title("Среднесуточный профиль мощности ветроустановки", fontsize=13, pad=15)
plt.xlabel("Час суток (UTC)", fontsize=12)
plt.ylabel("Средняя мощность, кВт", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("daily_power_profile.png", dpi=300)
plt.show()
