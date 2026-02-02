"""
microgrid_env_fixed.py

Исправленная версия Gymnasium-compatible окружения для микросети (PV + Wind + Battery).

Основные исправления:
1. Action space: одномерный continuous [-1, 1] для заряда/разряда
2. Правильная логика обработки действий с учётом знака
3. Улучшенная reward функция
4. Детальное логирование для отладки
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict


class MicrogridEnvFixed(gym.Env):
    """
    Окружение для управления микросетью с улучшенной логикой действий.
    
    Action space: Box([-1, 1], shape=(1,))
        - action > 0: заряд батареи (пропорционально max_charge_kw)
        - action < 0: разряд батареи (пропорционально max_discharge_kw)
        - action = 0: батарея не используется
    
    Логика работы:
        - При излишке (generation > load): можем заряжать батарею или сбрасывать
        - При дефиците (generation < load): можем разряжать батарею или допускать недопоставку
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        power_unit: str = "kW",
        forecast_horizon: int = 6,
        dt_hours: float = 1.0,
        battery_capacity_kwh: float = 50.0,
        soc_init: float = 0.5,
        max_charge_kw: float = 10.0,
        max_discharge_kw: float = 10.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        allow_grid_import: bool = False,
        w_unmet: float = 100.0,
        w_spill: float = 1.0,
        w_soc_violation: float = 500.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        # Проверка единиц мощности
        assert power_unit in ("W", "kW"), "power_unit должен быть 'W' или 'kW'"
        
        # Сохранение параметров
        self.raw_data = data.copy().sort_index()
        self.power_unit = power_unit
        self.mult = 1.0 if power_unit == "kW" else 1.0 / 1000.0
        self.horizon = forecast_horizon
        self.dt = dt_hours
        
        # Параметры батареи
        self.capacity = battery_capacity_kwh
        self.soc = soc_init * self.capacity
        self.soc_init = soc_init * self.capacity
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.eta_c = charge_efficiency
        self.eta_d = discharge_efficiency
        self.soc_min = soc_min * self.capacity
        self.soc_max = soc_max * self.capacity
        
        # Параметры штрафов
        self.allow_grid_import = allow_grid_import
        self.w_unmet = w_unmet
        self.w_spill = w_spill
        self.w_soc_violation = w_soc_violation
        
        # Рендеринг
        self.render_mode = render_mode
        self.seed(seed)
        
        # Проверка индекса данных
        if not isinstance(self.raw_data.index, pd.DatetimeIndex):
            raise ValueError("data должен иметь DatetimeIndex")
        
        # Проверка наличия необходимых колонок
        required_cols = ["solar_ac", "wind_ac", "load"]
        for col in required_cols:
            if col not in self.raw_data.columns:
                raise ValueError(f"data должен содержать колонку '{col}'")
        
        # Action space: одномерный continuous в [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: 24 измерения
        # [solar_now, wind_now, load_now, soc_now,
        #  solar_forecast[6], wind_forecast[6], load_forecast[6],
        #  hour_sin, hour_cos]
        obs_low = -np.inf * np.ones(24, dtype=np.float32)
        obs_high = np.inf * np.ones(24, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        # Внутренние указатели
        self._t_idx = None
        self._n_steps = len(self.raw_data)
        
        # Статистика
        self.cumulative_reward = 0.0
        self.cumulative_unmet = 0.0
        self.cumulative_spill = 0.0
        self.episode_steps = 0

    def _get_row(self, idx: int) -> Tuple[float, float, float]:
        """Получение значений solar, wind, load для индекса idx в кВт"""
        row = self.raw_data.iloc[idx]
        s = float(row["solar_ac"]) * self.mult
        w = float(row["wind_ac"]) * self.mult
        l = float(row["load"]) * self.mult
        return s, w, l

    def _build_state(self, idx: int) -> np.ndarray:
        """Построение вектора состояния"""
        # Текущие значения
        s_now, w_now, l_now = self._get_row(idx)
        soc_now = self.soc / self.capacity  # нормализованный SOC [0..1]
        
        # Прогнозы на следующие horizon часов
        s_fore = np.zeros(self.horizon, dtype=float)
        w_fore = np.zeros(self.horizon, dtype=float)
        l_fore = np.zeros(self.horizon, dtype=float)
        
        for h in range(1, self.horizon + 1):
            j = idx + h
            if j < self._n_steps:
                s_fore[h-1] = float(self.raw_data.iloc[j]["solar_ac"]) * self.mult
                w_fore[h-1] = float(self.raw_data.iloc[j]["wind_ac"]) * self.mult
                l_fore[h-1] = float(self.raw_data.iloc[j]["load"]) * self.mult
            # Если выходим за границы, остаются нули
        
        # Кодирование времени суток
        ts = self.raw_data.index[idx]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # Собираем вектор состояния
        vec = np.concatenate([
            np.array([s_now, w_now, l_now, soc_now], dtype=float),
            s_fore.astype(float),
            w_fore.astype(float),
            l_fore.astype(float),
            np.array([hour_sin, hour_cos], dtype=float)
        ])
        
        assert vec.shape[0] == 24, f"Неверная размерность состояния: {vec.shape[0]}"
        return vec.astype(np.float32)

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Сброс окружения"""
        super().reset(seed=seed)
        
        # Сброс статистики
        self.cumulative_reward = 0.0
        self.cumulative_unmet = 0.0
        self.cumulative_spill = 0.0
        self.episode_steps = 0
        
        # Выбор случайного начального индекса
        max_start = max(0, self._n_steps - self.horizon - 100)
        start_idx = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0
        self._t_idx = int(start_idx)
        
        # Сброс SOC
        self.soc = self.soc_init
        
        # Получение начального состояния
        obs = self._build_state(self._t_idx)
        info = {"t_idx": self._t_idx, "timestamp": self.raw_data.index[self._t_idx]}
        
        return obs, info

    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Выполнение одного шага в окружении.
        
        Логика обработки действий:
        - action[0] в [-1, 1]
        - action > 0: намерение зарядить батарею (при наличии излишка)
        - action < 0: намерение разрядить батарею (при наличии дефицита)
        """
        assert self._t_idx is not None, "Вызовите reset() перед step()"
        
        # Нормализация действия
        a = float(np.clip(action[0], -1.0, 1.0))
        
        # Получение текущих значений
        s_now_kW, w_now_kW, load_kW = self._get_row(self._t_idx)
        generation_kW = s_now_kW + w_now_kW
        
        # Чистый баланс: положительный = излишек, отрицательный = дефицит
        net_kW = generation_kW - load_kW
        
        # Инициализация переменных
        charge_kw = 0.0
        discharge_kw = 0.0
        spill_kw = 0.0
        unmet_kw = 0.0
        
        # === ЛОГИКА ОБРАБОТКИ ДЕЙСТВИЙ ===
        
        if net_kW > 0:
            # ИЗЛИШЕК ЭНЕРГИИ
            # action > 0: заряжаем батарею
            # action <= 0: не заряжаем (энергия сбрасывается)
            
            if a > 0:
                # Желаемая мощность заряда
                desired_charge = a * self.max_charge_kw
                
                # Доступное место в батарее (кВт·ч)
                soc_headroom_kwh = max(0.0, self.soc_max - self.soc)
                
                # Максимальная мощность заряда с учётом емкости
                max_charge_by_headroom_kw = soc_headroom_kwh / self.dt
                
                # Фактическая мощность заряда
                charge_kw = min(desired_charge, net_kW, max_charge_by_headroom_kw)
            
            # Остаток энергии сбрасывается
            spill_kw = max(0.0, net_kW - charge_kw)
            
        else:
            # ДЕФИЦИТ ЭНЕРГИИ
            # action < 0: разряжаем батарею
            # action >= 0: не разряжаем (дефицит остаётся)
            
            deficit_kw = -net_kW  # положительное значение дефицита
            
            if a < 0:
                # Желаемая мощность разряда
                desired_discharge = (-a) * self.max_discharge_kw
                
                # Доступная энергия в батарее (кВт·ч)
                avail_energy_kwh = max(0.0, self.soc - self.soc_min)
                
                # Максимальная мощность разряда с учётом SOC
                max_discharge_by_soc_kw = avail_energy_kwh / self.dt
                
                # Фактическая мощность разряда
                discharge_kw = min(desired_discharge, deficit_kw, max_discharge_by_soc_kw)
            
            # Недопоставленная энергия
            unmet_kw = max(0.0, deficit_kw - discharge_kw)
        
        # === ОБНОВЛЕНИЕ SOC ===
        
        charged_kwh = 0.0
        discharged_kwh = 0.0
        
        if charge_kw > 0:
            # Энергия, добавленная в батарею с учётом эффективности
            charged_kwh = charge_kw * self.dt * self.eta_c
            self.soc += charged_kwh
        
        if discharge_kw > 0:
            # Энергия, извлечённая из батареи (с учётом потерь)
            removed_kwh = discharge_kw * self.dt / self.eta_d
            self.soc -= removed_kwh
            discharged_kwh = discharge_kw * self.dt  # доставленная энергия
        
        # === ПРОВЕРКА НАРУШЕНИЙ SOC ===
        
        soc_violation_penalty = 0.0
        
        if self.soc < self.soc_min:
            violation_kwh = self.soc_min - self.soc
            soc_violation_penalty += violation_kwh * self.w_soc_violation
            self.soc = self.soc_min
        
        if self.soc > self.soc_max:
            violation_kwh = self.soc - self.soc_max
            soc_violation_penalty += violation_kwh * self.w_soc_violation
            self.soc = self.soc_max
        
        # === РАСЧЁТ НАГРАДЫ ===
        
        # Энергетические потери в кВт·ч
        spill_kwh = spill_kw * self.dt
        unmet_kwh = unmet_kw * self.dt
        
        # Награда: отрицательные штрафы за потери
        reward = -(
            unmet_kwh * self.w_unmet +
            spill_kwh * self.w_spill +
            soc_violation_penalty
        )
        
        # Бонус за поддержание SOC в среднем диапазоне (опционально)
        soc_normalized = self.soc / self.capacity
        if 0.4 <= soc_normalized <= 0.6:
            reward += 0.1  # небольшой бонус
        
        # === ОБНОВЛЕНИЕ СТАТИСТИКИ ===
        
        self.cumulative_reward += reward
        self.cumulative_unmet += unmet_kwh
        self.cumulative_spill += spill_kwh
        self.episode_steps += 1
        
        # === ПЕРЕХОД К СЛЕДУЮЩЕМУ ШАГУ ===
        
        self._t_idx += 1
        done = False
        
        if self._t_idx >= self._n_steps - 1:
            done = True
        
        # Следующее состояние
        if not done:
            obs = self._build_state(self._t_idx)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Информация
        info = {
            "t_idx": self._t_idx,
            "timestamp": self.raw_data.index[self._t_idx - 1],
            "generation_kW": generation_kW,
            "load_kW": load_kW,
            "net_kW": net_kW,
            "charge_kw": charge_kw,
            "discharge_kw": discharge_kw,
            "spill_kw": spill_kw,
            "unmet_kw": unmet_kw,
            "soc_kwh": self.soc,
            "soc_normalized": self.soc / self.capacity,
            "charged_kwh": charged_kwh,
            "discharged_kwh": discharged_kwh,
            "episode_steps": self.episode_steps
        }
        
        return obs, float(reward), bool(done), False, info

    def render(self):
        """Простой текстовый вывод состояния"""
        if self._t_idx is None:
            print("Окружение не инициализировано.")
            return
        
        ts = self.raw_data.index[self._t_idx]
        soc_pct = (self.soc / self.capacity) * 100
        
        print(f"[{ts}] SOC={self.soc:.2f} kWh ({soc_pct:.1f}%)")

    def seed(self, seed: Optional[int] = None):
        """Установка seed для генератора случайных чисел"""
        self.np_random, seed_val = gym.utils.seeding.np_random(seed)
        return [seed_val]