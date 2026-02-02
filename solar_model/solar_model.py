import numpy as np
import pandas as pd
import pvlib
from pvlib import pvsystem, modelchain, location, temperature
import matplotlib.pyplot as plt
from math import ceil
from tqdm import tqdm
from pvlib.temperature import sapm_cell, TEMPERATURE_MODEL_PARAMETERS

class SolarModel:
    """
    Реалистичная PV-модель (вариант B):
    - clear-sky (Ineichen)
    - SAPM/CEC module + inverter
    - cell temperature (sapm_celltemp)
    - синтетическая облачность (smooth random process)
    - масштабирование под требуемую установленную мощность (AC target_kW)
    """

    def __init__(
        self,
        latitude=51.6210400,
        longitude=73.1108200,
        timezone="Asia/Almaty",
        tilt=30,
        azimuth=180,
        target_kw=11.0,
        module_name="Canadian_Solar_CS6X_300M",  # пример из CEC database
        inverter_name="ABB__MICRO_0_25_I_OUTD_US_208",  # пример
        module_power_stc=330,  # W (если выбран модуль, замените по реальным данным)
        albedo=0.2,
        seed=42
    ):
        self.lat = latitude
        self.lon = longitude
        self.tz = timezone
        self.tilt = tilt
        self.azimuth = azimuth
        self.target_kw = target_kw
        self.module_name = module_name
        self.inverter_name = inverter_name
        self.module_power_stc = module_power_stc
        self.albedo = albedo
        self.rng = np.random.default_rng(seed)

        # Location object
        self.location = location.Location(latitude, longitude, tz=timezone)

        # retrieve databases (CEC)
        self.cec_modules = pvsystem.retrieve_sam('CECMod')  # may be large
        self.cec_inverters = pvsystem.retrieve_sam('cecinverter')

        # Choose module/inverter params if available, else fallback to simple STC scaling
        if module_name in self.cec_modules.index:
            self.module = self.cec_modules.loc[module_name]
            # If module contains 'STC' rating etc - but we'll still use module_power_stc parameter
        else:
            self.module = None

        if inverter_name in self.cec_inverters.index:
            self.inverter = self.cec_inverters.loc[inverter_name]
        else:
            self.inverter = None

        # compute number of modules (integer)
        needed_w = target_kw * 1000.0
        self.n_modules = int(ceil(needed_w / self.module_power_stc))
        self.system_stc_w = self.n_modules * self.module_power_stc

        # scale factor to get as close as possible to target (we'll scale final AC by this)
        self.scale_factor = needed_w / self.system_stc_w

    def _synthesize_cloudiness(self, times, mean_cloud=0.18, max_cloud=0.6, scale_hours=24):
        """
        Синтезирует плавную временную серию облачности (в диапазоне 0..max_cloud),
        используя случайный нормальный шум и скользящую медиану/сглаживание.
        mean_cloud ~ средний коэффициент потери (например, 0.18 -> 18%).
        scale_hours — параметр сглаживания (чем больше — крупнее облачность по времени).
        """
        n = len(times)
        # белый шум
        noise = self.rng.normal(loc=0.0, scale=0.25, size=n)
        # cumulative to produce correlated changes, then normalize
        cum = np.convolve(noise, np.ones(24)/24, mode='same')
        # scale into 0..max_cloud with mean approx mean_cloud
        # use sigmoid to bound [0,1] then * max_cloud
        sig = 1 / (1 + np.exp(-cum))
        raw = mean_cloud + (sig - 0.5) * (max_cloud - mean_cloud) * 2
        cloud = np.clip(raw, 0.0, max_cloud)
        # apply additional smoothing with rolling mean (by hours)
        cloud_series = pd.Series(cloud, index=times).rolling(window=max(1, int(scale_hours//1)), center=True, min_periods=1).mean()
        return cloud_series.values

    def generate(self, start="2024-01-01", end="2024-12-31", freq="1h", add_noise=True):
        """
        Генерация временного ряда. Возвращает DataFrame с колонками:
        ['ghi','dni','dhi','poa_global','temp_air','wind_speed','temp_cell','dc_power','ac_power','cloud_factor']
        """
        times = pd.date_range(start=start, end=end, freq=freq, tz=self.tz)
        # 1) clear-sky (Ineichen)
        clearsky = self.location.get_clearsky(times, model='ineichen')  # returns ghi, dni, dhi

        # 2) solar position
        solpos = self.location.get_solarposition(times)

        # 3) POA on tilted plane
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=self.azimuth,
            dni=clearsky['dni'],
            ghi=clearsky['ghi'],
            dhi=clearsky['dhi'],
            solar_zenith=solpos['zenith'],
            solar_azimuth=solpos['azimuth'],
            albedo=self.albedo
        )

        # 4) synthesize weather: air temp and wind_speed (simple seasonal + noise)
        # seasonal temperature: sine wave + daily variation
        day_of_year = times.dayofyear.values
        hour = times.hour.values + times.minute.values / 60.0
        # mean annual temp for location ~ 8C (adjust if necessary)
        mean_annual_temp = 8.0
        amp_season = 12.0
        temp_air = mean_annual_temp + amp_season * np.sin(2 * np.pi * (day_of_year - 172)/365.0) \
                   + 5.0 * np.sin(2 * np.pi * hour / 24.0) \
                   + self.rng.normal(0, 1.8, size=len(times))
        # wind speed: mean ~ 3-6 m/s with daily noise
        mean_wind = 3.5
        wind_speed = np.abs(mean_wind + 1.5 * np.sin(2*np.pi*(hour/24.0 + 0.1)) + self.rng.normal(0, 1.0, size=len(times)))

        # 5) cloudiness factor (0..max) and apply to POA (simple multiplicative attenuation)
        cloud_factor = self._synthesize_cloudiness(times, mean_cloud=0.18, max_cloud=0.7, scale_hours=8)
        # ensure night times unaffected (POA ~ 0 anyway), but cloud could reduce daytime
        poa_global = poa['poa_global'].values * (1.0 - cloud_factor)

        # 6) cell temperature via sapm_celltemp
        # use wind_speed and poa_global:
        params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        temp_cell = sapm_cell(poa_global, temp_air, wind_speed, **params)
        # temp_cell = pvlib.temperature.sapm_cell(poa_global, temp_air, wind_speed,
        #                                            model='open_rack_glass_glass')  # model choice

        # 7) DC power estimate: use simple effective efficiency model with temp coefficient
        # default module efficiency at STC approx:
        stc_eff = 0.18  # use ~18% baseline; if CEC module available, can use more accurate
        # temperature coefficient (power %/°C) -> convert to absolute factor: typical -0.004 / °C
        gamma_p = -0.004  # per °C
        # reference cell temp at STC ~ 25C
        temp_ref = 25.0
        eff_temp_factor = 1.0 + gamma_p * (temp_cell - temp_ref)
        # ensure non-negative
        eff_temp_factor = np.clip(eff_temp_factor, 0.5, 1.2)
        dc_power_per_m2 = poa_global * stc_eff * eff_temp_factor  # W per m2 * module area? We'll scale via STC

        # Instead of computing module area, use module_power_stc and STC irradiance 1000 W/m2:
        # nominal power per module = module_power_stc (W) at 1000 W/m2 and 25C
        # so instantaneous DC per module approx = module_power_stc * (poa_global / 1000) * eff_temp_factor
        dc_power_per_module = self.module_power_stc * (poa_global / 1000.0) * eff_temp_factor
        total_dc = dc_power_per_module * self.n_modules  # W

        # 8) inverter: simple efficiency curve or pvlib inverter if available
        # Use pvlib.pvsystem.snlinverter (CEC inverter) if we have parameters - but for simplicity:
        # approximate inverter efficiency 98% near nominal, lower at low power.
        def inverter_efficiency(p_ac, p_nom):
            # p_ac and p_nom in W
            # approximate S-curve behavior: low eff at low p, peak ~0.98 at >20% rated
            frac = p_ac / max(1e-9, p_nom)
            eff = 0.9 + 0.08 * (1 - np.exp(-5 * frac))  # from 0.9 -> ~0.98
            return np.clip(eff, 0.75, 0.99)

        # estimate inverter nominal power as slightly above target_kW (1.1x)
        inverter_nom_w = self.target_kw * 1000.0 * 1.05
        inv_eff = inverter_efficiency(total_dc, inverter_nom_w)
        ac_power = total_dc * inv_eff

        # scale final AC to exactly target_kw if desired (to match requested capacity)
        # Many systems have some rounding due to module integer count; scale factor accounts for that:
        ac_power_scaled = ac_power * self.scale_factor

        # 9) Compose DataFrame
        df = pd.DataFrame(index=times)
        df['ghi'] = clearsky['ghi']
        df['dni'] = clearsky['dni']
        df['dhi'] = clearsky['dhi']
        df['poa_global'] = poa_global
        df['temp_air'] = temp_air
        df['wind_speed'] = wind_speed
        df['temp_cell'] = temp_cell
        df['dc_power'] = total_dc
        df['ac_power'] = ac_power_scaled
        df['cloud_factor'] = cloud_factor
        df['n_modules'] = self.n_modules
        df['system_stc_w'] = self.system_stc_w

        return df

    def plot_summary(self, df, period='7D'):
        """
        Быстрая визуализация: суточный профиль, годовая, распределение облачности.
        period — панель для примера (например '7D' или '365D')
        """
        # Annual generation (kWh per day)
        df_daily = df.resample('D').sum()
        annual_kwh = df_daily['ac_power'].sum() / 1000.0
        print(f"Estimated annual generation (kWh): {annual_kwh:.1f} kWh (approx)")

        # Plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        # 1) sample window
        sample = df.last(period)
        sample['ac_power'].plot(ax=axes[0], title=f"AC power (last {period}) [W]")
        axes[0].set_ylabel('W')

        # 2) typical day (mean by hour)
        hourly = df.groupby(df.index.hour)['ac_power'].mean()
        hourly.plot(ax=axes[1], title='Average daily profile (hourly mean)')
        axes[1].set_xlabel('Hour')
        axes[1].set_ylabel('W')

        # 3) cloud factor histogram
        axes[2].hist(df['cloud_factor'].dropna(), bins=30)
        axes[2].set_title('Cloud factor distribution')
        axes[2].set_xlabel('cloud factor')
        axes[2].set_ylabel('count')

        plt.tight_layout()
        plt.show()

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
