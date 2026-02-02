import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from scipy.special import gamma as gamma_func

class WindModel:
    """
    WindModel variant B (recommended):
    - base mean wind: seasonal (annual) + diurnal + noise
    - Weibull sampling for instantaneous fluctuations (shape k, scale lambda)
    - vertical shear (power law) to hub height
    - physical P = 0.5 * rho * A * Cp * v^3 but clipped by power curve (cut-in, rated, cut-out)
    - simple generator/converter efficiency
    - returns DataFrame with columns:
      ['wind_speed_ref', 'wind_speed_hub', 'power_raw_W', 'power_ac_W', 'rho', 'turbine_status']
    """
    def __init__(
        self,
        reference_height=10.0,   # m - reference measurement height
        hub_height=50.0,         # m - turbine hub height
        shear_exp=0.14,          # wind shear exponent (0.12-0.2 typical)
        mean_wind_annual=5.0,    # m/s average at ref height (can be adapted)
        weibull_k=2.0,           # Weibull shape
        rotor_diameter=13.0,     # m -> area ~ pi*r^2, choose for 50 kW class (example)
        rated_power_kw=50.0,     # kW rated power per turbine
        cut_in=3.5,              # m/s
        rated_wind=12.0,         # m/s
        cut_out=25.0,            # m/s
        cp=0.42,                 # power coefficient (Cp), typical 0.3-0.45
        efficiency=0.95,         # generator+converter efficiency
        air_density_sea_level=1.225, # kg/m3 at 15°C sea level
        temp_amplitude=10.0,     # amplitude for seasonal temp oscillation (for rho variation)
        seed=12345
    ):
        self.ref_h = reference_height
        self.hub_h = hub_height
        self.alpha = shear_exp
        self.mean_wind_annual = mean_wind_annual
        self.k = weibull_k
        self.D = rotor_diameter
        self.A = pi * (rotor_diameter/2.0)**2
        self.rated_w = rated_wind
        self.rated_power_w = rated_power_kw * 1000.0
        self.cut_in = cut_in
        self.cut_out = cut_out
        self.cp = cp
        self.eta = efficiency
        self.rho0 = air_density_sea_level
        self.temp_amp = temp_amplitude
        self.rng = np.random.default_rng(seed)

        # safety: ensure physical correspondence between rotor and rated power (user can override)
        # We'll not rescale rotor, but power curve will cap at rated power.

    def _weibull_lambda_from_mean(self, mean, k):
        """
        For Weibull distribution: mean = lambda * Gamma(1 + 1/k)
        -> lambda = mean / Gamma(1 + 1/k)
        """
        return mean / gamma_func(1.0 + 1.0 / k)

    def _sample_weibull(self, scale, k, size):
        """Sample Weibull with given scale (lambda) and shape k"""
        # numpy weibull uses form: draws from Weibull(k) and scales by lambda
        # but numpy.random.weibull(k) returns samples of Weibull with scale=1, shape=k
        samples = self.rng.weibull(k, size=size) * scale
        return samples

    def _power_curve(self, v_hub):
        """
        Map hub wind speed to turbine electrical AC power (W) by piecewise power curve:
         - v < cut_in -> 0
         - cut_in <= v < rated -> cubic (physics) scaled to rated
         - rated <= v < cut_out -> rated_power
         - v >= cut_out -> 0 (stop)
        """
        p = np.zeros_like(v_hub)
        # region between cut_in and rated: use cubic scaling to match rated at rated_w
        mask_oper = (v_hub >= self.cut_in) & (v_hub < self.rated_w)
        # physical cubic proportionality constant to match rated at rated_w:
        # compute P_physical = 0.5 * rho * A * Cp * v^3 * eta, but that may exceed rated -> we'll scale
        # Instead derive a cube law factor then scale to hit rated at rated_w
        # so p = rated_power * (v/rated_w)^3 for v in [cut_in, rated_w)
        p[mask_oper] = self.rated_power_w * (v_hub[mask_oper] / self.rated_w) ** 3
        # rated region
        mask_rated = (v_hub >= self.rated_w) & (v_hub < self.cut_out)
        p[mask_rated] = self.rated_power_w
        # outside (cut-in below and cut-out above) remain zero
        # optionally apply generator efficiency factor, but rated_power already is electrical.
        # ensure not to exceed rated (numerical)
        p = np.clip(p, 0.0, self.rated_power_w)
        return p

    def generate(self, start="2024-01-01", end="2024-12-31", freq="1h", add_gusts=True):
        """
        Generate wind time series and power for given period and frequency.
        Returns DataFrame with:
         - wind_speed_ref: m/s at reference height (ref_h)
         - wind_speed_hub: m/s at hub height (after shear)
         - power_raw_W: raw physical power 0.5*rho*A*Cp*v^3 (W)
         - power_ac_W: after power curve and efficiency (W)
         - rho: air density used (kg/m3)
         - turbine_status: string (stopped/operational)
        """
        times = pd.date_range(start=start, end=end, freq=freq, tz="Asia/Almaty")
        n = len(times)

        # 1) create seasonal + diurnal baseline mean wind at reference height
        doy = times.dayofyear.values
        hour = times.hour.values + times.minute.values/60.0

        # seasonal mean: shift to peak roughly in spring/autumn depending on locale -> use sine with peak ~ day 80 (Mar)
        seasonal = 0.8 * np.sin(2 * np.pi * (doy - 100) / 365.0)  # amplitude ~0.8 m/s
        diurnal = 0.6 * np.sin(2 * np.pi * (hour - 15) / 24.0)    # amplitude daily ~0.6 m/s peak ~15h
        baseline = self.mean_wind_annual + seasonal + diurnal

        # 2) map baseline -> weibull scale (lambda) for each timestep using mean->lambda conversion
        lambda_t = self._weibull_lambda_from_mean(np.clip(baseline, 0.1, None), self.k)

        # 3) sample instantaneous winds from Weibull (correlated smoothing)
        raw_samples = self._sample_weibull(scale=lambda_t, k=self.k, size=n)

        # Introduce temporal correlation: low-pass filter (exponential smoothing)
        alpha_corr = 0.6  # smoothing factor
        v_ref = np.empty(n)
        v_ref[0] = raw_samples[0]
        for i in range(1, n):
            v_ref[i] = alpha_corr * raw_samples[i] + (1 - alpha_corr) * v_ref[i-1]

        # 4) gusts: occasional random spikes (Poisson arrivals)
        if add_gusts:
            # average gusts per day
            lambda_gusts_per_day = 0.3  # about 0.3 gust events per day
            prob_per_step = lambda_gusts_per_day * ( (pd.to_timedelta(freq).total_seconds()) / 86400.0 )
            gust_mask = self.rng.random(n) < prob_per_step
            # gust magnitude factor (1.2..1.8)
            gust_mags = 1.0 + self.rng.random(n) * 1.0  # up to +100%
            v_ref = v_ref * np.where(gust_mask, gust_mags, 1.0)

        # 5) vertical shear to hub height: v_hub = v_ref * (hub / ref)^alpha
        hub_factor = (self.hub_h / self.ref_h) ** self.alpha
        v_hub = v_ref * hub_factor

        # 6) air density: simple seasonal/temperature proxy (colder -> higher density)
        # approximate temperature as seasonal sine; density approx rho0 * (1 - 0.00366*(T-15))
        # so use sinusoidal temp with mean 15C and amplitude temp_amp
        T_mean = 15.0
        T = T_mean + self.temp_amp * np.sin(2 * np.pi * (doy - 200) / 365.0)
        rho = self.rho0 * (1 - 0.00366 * (T - 15.0))

        # 7) raw physical power (W) using P = 0.5 * rho * A * Cp * v^3 (but note: this can exceed rated)
        power_physical = 0.5 * rho * self.A * self.cp * (v_hub ** 3)

        # 8) apply power curve to get electrical power (AC)
        power_curve_limited = self._power_curve(v_hub)
        # the power curve returns electrical at rated; apply simple efficiency factor for generator losses
        power_ac = power_curve_limited * self.eta

        # set turbine status
        status = np.where((v_hub >= self.cut_in) & (v_hub < self.cut_out), 'operational', 'stopped')

        # Build dataframe
        df = pd.DataFrame(index=times)
        df['wind_speed_ref'] = v_ref
        df['wind_speed'] = v_hub  # alias 'wind_speed' to match other models expecting hub speed
        df['power_raw_W'] = power_physical
        df['power_curve_W'] = power_curve_limited
        df['power_ac_W'] = power_ac
        df['rho'] = rho
        df['turbine_status'] = status
        df['rotor_diameter_m'] = self.D
        df['rated_power_W'] = self.rated_power_w
        return df

    def plot_summary(self, df, period='7D'):
        """
        Quick visualization: sample window, average daily profile, pdf of speeds
        """
        df_daily = df.resample('D').sum()
        annual_mwh = df_daily['power_ac_W'].sum() / 1e6
        print(f"Estimated annual generation (MWh): {annual_mwh:.3f} MWh (approx)")

        fig, axes = plt.subplots(3,1,figsize=(10,10))
        sample = df.last(period)
        sample['power_ac_W'].plot(ax=axes[0], title=f"AC power (last {period}) [W]")
        axes[0].set_ylabel('W')

        # mean daily profile (hourly mean)
        hourly = df.groupby(df.index.hour)['power_ac_W'].mean()
        hourly.plot(ax=axes[1], title='Average daily power profile (hourly mean)')
        axes[1].set_xlabel('Hour')

        # wind speed histogram
        axes[2].hist(df['wind_speed'].dropna(), bins=40)
        axes[2].set_title('Wind speed (hub) distribution')
        axes[2].set_xlabel('m/s')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = WindModel(
        reference_height=10.0,
        hub_height=50.0,
        shear_exp=0.14,
        mean_wind_annual=4.5,  # set according to site (try 4.5..6.0 m/s)
        weibull_k=2.0,
        rotor_diameter=13.0,
        rated_power_kw=50.0,
        cut_in=3.5,
        rated_wind=12.0,
        cut_out=25.0,
        cp=0.42,
        efficiency=0.95
    )
    df = model.generate(start="2024-01-01", end="2024-12-31", freq="1h", add_gusts=True)
    df.to_csv('wind_df.csv')
    model.plot_summary(df, period='14D')
