"""
Улучшенный скрипт обучения PPO агента для управления микросетью.

Основные улучшения:
1. Правильная генерация данных с использованием SolarModel и WindModel
2. Добавление forecast-based признаков в состояние
3. Корректная обработка action space
4. Улучшенная визуализация результатов
5. Сохранение метрик обучения
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sys

# Добавляем пути к модулям
sys.path.insert(0, '/mnt/user-data/uploads')

from microgrid_env import MicrogridEnvFixed
from solar_model.solar_model import SolarModel
from wind_model.wind_model import WindModel


class MetricsCallback(BaseCallback):
    """Callback для сохранения метрик во время обучения"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Проверяем завершение эпизода
        if self.locals.get('dones')[0]:
            # Получаем информацию об эпизоде
            info = self.locals.get('infos')[0]
            self.episode_count += 1
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count} finished")
        
        return True


def generate_synthetic_load(times, base_load_kw=6.0, peak_load_kw=12.0):
    """
    Генерирует синтетическую нагрузку с суточной и недельной периодичностью.
    
    Args:
        times: DatetimeIndex
        base_load_kw: базовая нагрузка в кВт
        peak_load_kw: пиковая нагрузка в кВт
    
    Returns:
        np.array: массив значений нагрузки в кВт
    """
    n = len(times)
    
    # Суточная компонента (пик днём)
    hour = times.hour.values + times.minute.values / 60.0
    daily_pattern = 0.5 * (1 + np.sin(2 * np.pi * (hour - 6) / 24.0))
    
    # Недельная компонента (выше в будние дни)
    day_of_week = times.dayofweek.values
    weekly_pattern = np.where(day_of_week < 5, 1.1, 0.9)  # будни vs выходные
    
    # Сезонная компонента
    day_of_year = times.dayofyear.values
    seasonal_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * (day_of_year - 172) / 365.0)
    
    # Комбинируем паттерны
    load = base_load_kw + (peak_load_kw - base_load_kw) * daily_pattern
    load = load * weekly_pattern * seasonal_pattern
    
    # Добавляем шум
    noise = np.random.normal(0, 0.3, size=n)
    load = load + noise
    
    return np.maximum(load, 0.5)  # минимальная нагрузка 0.5 кВт


def prepare_training_data(
    start_date="2024-01-01",
    end_date="2024-03-31",
    freq="1h",
    solar_target_kw=11.0,
    wind_rated_kw=50.0
):
    """
    Подготовка данных для обучения с использованием физических моделей.
    
    Args:
        start_date: начальная дата
        end_date: конечная дата
        freq: частота дискретизации
        solar_target_kw: установленная мощность солнечной станции (кВт)
        wind_rated_kw: номинальная мощность ветрогенератора (кВт)
    
    Returns:
        pd.DataFrame: датафрейм с колонками solar_ac, wind_ac, load
    """
    print("🔄 Генерация данных солнечной станции...")
    solar_model = SolarModel(
        latitude=51.6210400,
        longitude=73.1108200,
        timezone="Asia/Almaty",
        tilt=30,
        azimuth=180,
        target_kw=solar_target_kw,
        module_power_stc=330,
        seed=42
    )
    solar_df = solar_model.generate(start=start_date, end=end_date, freq=freq)
    
    print("🔄 Генерация данных ветрогенератора...")
    wind_model = WindModel(
        reference_height=10.0,
        hub_height=50.0,
        mean_wind_annual=4.8,
        rotor_diameter=13.0,
        rated_power_kw=wind_rated_kw,
        cut_in=3.5,
        rated_wind=12.0,
        cut_out=25.0,
        cp=0.42,
        efficiency=0.95,
        seed=12345
    )
    wind_df = wind_model.generate(start=start_date, end=end_date, freq=freq)
    
    print("🔄 Генерация профиля нагрузки...")
    times = pd.date_range(start=start_date, end=end_date, freq=freq, tz="Asia/Almaty")
    load = generate_synthetic_load(times, base_load_kw=5.0, peak_load_kw=15.0)
    
    # Объединяем данные
    df = pd.DataFrame(index=times)
    df['solar_ac'] = solar_df['ac_power'].values / 1000.0  # конвертируем в кВт
    df['wind_ac'] = wind_df['power_ac_W'].values / 1000.0  # конвертируем в кВт
    df['load'] = load
    
    print(f"✅ Данные подготовлены: {len(df)} временных точек")
    print(f"   Солнечная генерация: {df['solar_ac'].mean():.2f} кВт (среднее)")
    print(f"   Ветровая генерация: {df['wind_ac'].mean():.2f} кВт (среднее)")
    print(f"   Нагрузка: {df['load'].mean():.2f} кВт (среднее)")
    
    return df


def train_ppo_agent(
    df_train,
    battery_capacity_kwh=50.0,
    max_charge_kw=10.0,
    max_discharge_kw=10.0,
    total_timesteps=500_000,
    save_dir="results/ppo_training"
):
    """
    Обучение PPO агента.
    
    Args:
        df_train: DataFrame с данными для обучения
        battery_capacity_kwh: емкость батареи в кВт·ч
        max_charge_kw: максимальная мощность заряда в кВт
        max_discharge_kw: максимальная мощность разряда в кВт
        total_timesteps: общее количество шагов обучения
        save_dir: директория для сохранения результатов
    
    Returns:
        trained model, metrics
    """
    # Создаём директорию для результатов
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\n🏗️  Создание окружения...")
    env = MicrogridEnvFixed(
        data=df_train,
        power_unit="kW",
        forecast_horizon=6,
        dt_hours=1.0,
        battery_capacity_kwh=battery_capacity_kwh,
        soc_init=0.5,
        max_charge_kw=max_charge_kw,
        max_discharge_kw=max_discharge_kw,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        soc_min=0.10,
        soc_max=0.90,
        allow_grid_import=False,
        w_unmet=100.0,  # высокий штраф за недопоставку
        w_spill=1.0,    # низкий штраф за curtailment
        w_soc_violation=500.0,  # очень высокий штраф за нарушение SOC
        seed=42
    )
    
    # Оборачиваем в векторное окружение
    vec_env = DummyVecEnv([lambda: env])
    
    print("🤖 Инициализация PPO агента...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,  # количество шагов для сбора данных
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # коэффициент энтропии для исследования
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
        device='cpu',
        tensorboard_log=str(save_path / "tensorboard")
    )
    
    print(f"\n🚀 Начало обучения на {total_timesteps} шагов...")
    
    # Callback для метрик
    callback = MetricsCallback(verbose=1)
    
    # Обучение
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Сохранение модели
    model_path = save_path / "ppo_microgrid_model"
    model.save(str(model_path))
    print(f"\n✅ Модель сохранена: {model_path}")
    
    return model, callback


def evaluate_agent(model, df_test, battery_capacity_kwh=50.0, save_dir="results/ppo_training"):
    """
    Оценка обученного агента на тестовых данных.
    
    Args:
        model: обученная модель PPO
        df_test: тестовый датасет
        battery_capacity_kwh: емкость батареи
        save_dir: директория для сохранения результатов
    """
    save_path = Path(save_dir)
    
    print("\n📊 Оценка агента на тестовых данных...")
    
    # Создаём тестовое окружение
    env_test = MicrogridEnvFixed(
        data=df_test,
        power_unit="kW",
        forecast_horizon=6,
        battery_capacity_kwh=battery_capacity_kwh,
        max_charge_kw=10.0,
        max_discharge_kw=10.0,
        w_unmet=100.0,
        w_spill=1.0,
        w_soc_violation=500.0,
        seed=42
    )
    
    # Тестирование
    obs, info = env_test.reset()
    
    # Списки для сохранения истории
    rewards = []
    actions = []
    socs = []
    generation = []
    load_vals = []
    unmet = []
    spill = []
    charge_power = []
    discharge_power = []
    
    done = False
    step_count = 0
    
    while not done and step_count < len(df_test):
        # Предсказание действия
        action, _ = model.predict(obs, deterministic=True)
        
        # Выполнение шага
        obs, reward, done, truncated, info = env_test.step(action)
        
        # Сохранение данных
        rewards.append(reward)
        actions.append(action[0])
        socs.append(info['soc_kwh'])
        generation.append(info['generation_kW'])
        load_vals.append(df_test.iloc[step_count]['load'])
        unmet.append(info['unmet_kw'])
        spill.append(info['spill_kw'])
        charge_power.append(info['charge_kw'])
        discharge_power.append(info['discharge_kw'])
        
        step_count += 1
    
    print(f"✅ Тестирование завершено: {step_count} шагов")
    print(f"   Средняя награда: {np.mean(rewards):.2f}")
    print(f"   Суммарная недопоставка: {sum(unmet):.2f} кВт·ч")
    print(f"   Суммарный curtailment: {sum(spill):.2f} кВт·ч")
    
    # Создание графиков
    create_evaluation_plots(
        df_test.index[:step_count],
        rewards, actions, socs, generation, load_vals,
        unmet, spill, charge_power, discharge_power,
        save_path
    )
    
    # Сохранение результатов в CSV
    results_df = pd.DataFrame({
        'reward': rewards,
        'action': actions,
        'soc_kwh': socs,
        'generation_kW': generation,
        'load_kW': load_vals,
        'unmet_kW': unmet,
        'spill_kW': spill,
        'charge_kW': charge_power,
        'discharge_kW': discharge_power
    }, index=df_test.index[:step_count])
    
    results_df.to_csv(save_path / 'evaluation_results.csv')
    print(f"📁 Результаты сохранены: {save_path / 'evaluation_results.csv'}")
    
    return results_df


def create_evaluation_plots(times, rewards, actions, socs, generation, load_vals,
                           unmet, spill, charge_power, discharge_power, save_path):
    """Создание графиков для оценки работы агента"""
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    
    # 1. Награды
    axes[0].plot(times, rewards, color='blue', alpha=0.7)
    axes[0].set_title('Награда по времени', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Награда')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 2. Действия
    axes[1].plot(times, actions, color='green', alpha=0.7)
    axes[1].set_title('Действия агента', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Действие')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. SOC батареи
    axes[2].plot(times, socs, color='purple', linewidth=2)
    axes[2].set_title('Состояние заряда батареи (SOC)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('SOC (кВт·ч)')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Баланс энергии
    axes[3].plot(times, generation, label='Генерация', color='orange', linewidth=1.5)
    axes[3].plot(times, load_vals, label='Нагрузка', color='red', linewidth=1.5)
    axes[3].fill_between(times, 0, generation, alpha=0.2, color='orange')
    axes[3].fill_between(times, 0, load_vals, alpha=0.2, color='red')
    axes[3].set_title('Баланс генерации и нагрузки', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Мощность (кВт)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    # 5. Заряд/разряд батареи и потери
    axes[4].plot(times, charge_power, label='Заряд', color='green', linewidth=1.5)
    axes[4].plot(times, [-d for d in discharge_power], label='Разряд', color='blue', linewidth=1.5)
    axes[4].plot(times, unmet, label='Недопоставка', color='red', linewidth=1.5, linestyle='--')
    axes[4].plot(times, spill, label='Curtailment', color='gray', linewidth=1.5, linestyle=':')
    axes[4].set_title('Работа батареи и потери энергии', fontsize=12, fontweight='bold')
    axes[4].set_ylabel('Мощность (кВт)')
    axes[4].set_xlabel('Время')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
    print(f"📊 Графики сохранены: {save_path / 'evaluation_plots.png'}")
    plt.close()
    
    # Дополнительный график: распределения
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Распределение наград
    axes[0, 0].hist(rewards, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Распределение наград')
    axes[0, 0].set_xlabel('Награда')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Распределение действий
    axes[0, 1].hist(actions, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Распределение действий')
    axes[0, 1].set_xlabel('Действие')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Распределение SOC
    axes[1, 0].hist(socs, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Распределение SOC')
    axes[1, 0].set_xlabel('SOC (кВт·ч)')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Баланс энергии (генерация - нагрузка)
    net_power = [g - l for g, l in zip(generation, load_vals)]
    axes[1, 1].hist(net_power, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Распределение чистой мощности')
    axes[1, 1].set_xlabel('Генерация - Нагрузка (кВт)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'distributions.png', dpi=150, bbox_inches='tight')
    print(f"📊 Распределения сохранены: {save_path / 'distributions.png'}")
    plt.close()


def main():
    """Основная функция для обучения и оценки PPO агента"""
    
    print("=" * 70)
    print("PPO АГЕНТ ДЛЯ УПРАВЛЕНИЯ МИКРОСЕТЬЮ")
    print("=" * 70)
    
    # Параметры
    TRAIN_START = "2024-01-01"
    TRAIN_END = "2024-02-29"  # 2 месяца для обучения
    TEST_START = "2024-03-01"
    TEST_END = "2024-03-31"  # 1 месяц для тестирования
    
    BATTERY_CAPACITY = 50.0  # кВт·ч
    MAX_CHARGE = 10.0  # кВт
    MAX_DISCHARGE = 10.0  # кВт
    TOTAL_TIMESTEPS = 300_000
    
    SAVE_DIR = "results/ppo_training"
    
    # 1. Подготовка данных
    print("\n" + "=" * 70)
    print("ЭТАП 1: ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)
    
    df_train = prepare_training_data(
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        solar_target_kw=11.0,
        wind_rated_kw=50.0
    )
    
    df_test = prepare_training_data(
        start_date=TEST_START,
        end_date=TEST_END,
        solar_target_kw=11.0,
        wind_rated_kw=50.0
    )
    
    # Сохранение данных
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(save_path / 'train_data.csv')
    df_test.to_csv(save_path / 'test_data.csv')
    
    # 2. Обучение агента
    print("\n" + "=" * 70)
    print("ЭТАП 2: ОБУЧЕНИЕ PPO АГЕНТА")
    print("=" * 70)
    
    model, callback = train_ppo_agent(
        df_train=df_train,
        battery_capacity_kwh=BATTERY_CAPACITY,
        max_charge_kw=MAX_CHARGE,
        max_discharge_kw=MAX_DISCHARGE,
        total_timesteps=TOTAL_TIMESTEPS,
        save_dir=SAVE_DIR
    )
    
    # 3. Оценка агента
    print("\n" + "=" * 70)
    print("ЭТАП 3: ОЦЕНКА АГЕНТА НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 70)
    
    results_df = evaluate_agent(
        model=model,
        df_test=df_test,
        battery_capacity_kwh=BATTERY_CAPACITY,
        save_dir=SAVE_DIR
    )
    
    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ И ОЦЕНКА ЗАВЕРШЕНЫ")
    print("=" * 70)
    print(f"📁 Все результаты сохранены в: {SAVE_DIR}")


if __name__ == "__main__":
    main()