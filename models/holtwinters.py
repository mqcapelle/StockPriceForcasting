from pathlib import Path
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Optional
import matplotlib.pyplot as plt

from utils.general_utils import load_config


class HoltWintersModel:
    def __init__(self, data: pd.DataFrame, config_path: Path = Path("config/settings.yaml"),
                 target_col: str = 'Close',
                 trend: Optional[str] = 'add',
                 seasonal: Optional[str] = None,
                 seasonal_periods: Optional[int] = None,
                 resampling_freq: str = 'B'):
        self.config = load_config(config_path)
        self.target_col = target_col
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.resampling_freq = resampling_freq

        self.series = self._extract_series(data, resampling_freq)
        self.model_fit: Optional[ExponentialSmoothing] = None

    def _extract_series(self, data: pd.DataFrame, resampling_freq: str) -> pd.Series:
        try:
            series = data[self.target_col]
            resampled = series.resample(resampling_freq).mean()
            return resampled.dropna()
        except KeyError:
            raise KeyError(f"'{self.target_col}' not found in DataFrame.")

    def fit(self):
        self.model_fit = ExponentialSmoothing(
            self.series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated"
        ).fit()
        print(f"[INFO] Holt-Winters model fitted with trend='{self.trend}', seasonal='{self.seasonal}'")

    def forecast(self, steps: int) -> pd.Series:
        if self.model_fit is None:
            raise RuntimeError("Model not fitted yet. Call fit().")

        forecast = self.model_fit.forecast(steps)
        forecast.index = pd.date_range(
            start=self.series.index[-1] + pd.tseries.offsets.BusinessDay(1),
            periods=steps,
            freq=self.resampling_freq
        )
        return forecast

    def plot_forecast(self, ax=None, steps: int = 10, include_conf_int: bool = False):
        if self.model_fit is None:
            raise RuntimeError("Model not fitted yet. Call fit().")

        forecast = self.forecast(steps)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.series, label='Historical', color='black')
        ax.plot(forecast.index, forecast.values, label='Forecast', color='blue')

        ax.axvline(self.series.index[-1], color='gray', linestyle='--', label='Forecast Start')
        ax.set_title(f"Holt-Winters Forecast for {self.target_col}")
        ax.set_xlabel("Date")
        ax.set_ylabel(self.target_col)
        ax.grid(True)
        ax.legend()

        return ax


# Example usage
if __name__ == "__main__":
    from utils.data_loader import StockDataLoader

    loader = StockDataLoader("config/settings.yaml")
    df = loader.load_data()

    model = HoltWintersModel(
        df,
        target_col="Close",
        trend='mul',
        seasonal='mul',  # 'add' or 'mul'
        seasonal_periods=52,  # step size, if using seasonal component
        resampling_freq='W'
    )

    model.fit()
    forecast = model.forecast(steps=10)

    ax = model.plot_forecast(steps=10)
    # ax.set_xlim(left=19700)
    # ax.set_ylim(bottom=175)
    plt.show()
