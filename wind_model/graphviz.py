import matplotlib.pyplot as plt
import pandas as pd

# Загрузи данные, если они сохранены
df = pd.read_csv("wind_df.csv", parse_dates=["Unnamed: 0"], index_col="Unnamed: 0")

plt.figure(figsize=(10,6))
plt.hist(df["wind_speed"], bins=40, color="#2E86AB", edgecolor="black", alpha=0.8)

plt.title("Распределение скоростей ветра на высоте ступицы турбины, м/с", fontsize=13, pad=15)
plt.xlabel("Скорость ветра, м/с", fontsize=12)
plt.ylabel("Частота наблюдений", fontsize=12)
plt.grid(alpha=0.3)

# Добавим текстовую подпись с модой
mode_speed = df["wind_speed"].mode()[0]
plt.axvline(mode_speed, color='red', linestyle='--', label=f'Мода ≈ {mode_speed:.1f} м/с')
plt.legend()

plt.tight_layout()
plt.savefig("wind_speed_distribution.png", dpi=300)
plt.show()
