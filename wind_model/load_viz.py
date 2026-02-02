import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Временной индекс
times = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="1h", tz="Asia/Almaty")

# 2️⃣ Суточные и сезонные переменные (теперь — numpy)
doy = times.dayofyear.to_numpy()
hour = (times.hour + times.minute / 60.0).to_numpy()

# 3️⃣ Суточные пики (утро и вечер)
morning = 0.3 * np.exp(-((hour - 9)**2) / 8)
evening = 0.7 * np.exp(-((hour - 20)**2) / 12)
daily_profile = 8000 * (morning + evening)

# 4️⃣ Сезонная модуляция
seasonal = 1.0 + 0.35 * np.sin(2 * np.pi * (doy - 320) / 365)

# 5️⃣ Комбинируем
load_base = daily_profile * seasonal

# 6️⃣ Добавляем шум
noise = np.random.normal(0, 0.05 * load_base.mean(), size=len(times))

# 7️⃣ Итоговая нагрузка
load = np.clip(load_base + noise, 1000, 20000)

# 8️⃣ Визуализация
plt.figure(figsize=(14, 5))
plt.plot(times, load / 1000, color="#2E86C1", lw=1)
plt.title("Профиль нагрузки автономной энергосистемы (2024 год)", fontsize=13, pad=15)
plt.ylabel("Нагрузка, кВт", fontsize=12)
plt.xlabel("Дата", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
