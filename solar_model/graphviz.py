import os

from solar_model import SolarModel
import matplotlib.pyplot as plt

def plot_all_columns(df, output_dir="solar_graphs"):
    """
    Строит и сохраняет графики по всем столбцам DataFrame,
    создаваемого SolarModel.generate().
    Каждый график сохраняется в формате JPEG.
    """
    # создаём папку для графиков, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # проходим по всем столбцам DataFrame
    for col in df.columns:
        plt.figure(figsize=(10, 4))
        df[col].plot(title=f"{col} over time", linewidth=1.0, color="tab:blue")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # путь к файлу
        filepath = os.path.join(output_dir, f"{col}.jpeg")

        # сохраняем в JPEG с хорошим качеством
        plt.savefig(filepath, format="jpeg", dpi=200)
        plt.close()

    print(f"✅ Все графики сохранены в папку: {os.path.abspath(output_dir)}")

# ---------------------------
# Пример использования
# ---------------------------
if __name__ == "__main__":
    model = SolarModel(
        latitude=51.6210400,
        longitude=73.1108200,
        timezone="Asia/Almaty",
        tilt=30,
        azimuth=180,
        target_kw=11.0,
        module_power_stc=330
    )

    df = model.generate(start="2024-01-01", end="2024-12-31", freq="1h")
    df.to_csv('solar_df.csv')
    model.plot_summary(df, period='14D')

    # строим графики по всем столбцам и сохраняем
    plot_all_columns(df, output_dir="solar_graphs")
