from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from utils.general_utils import load_config


class SimpleExpSmoothingModel:
    """
    Wrapper for Simple Exponential Smoothing (SES) model.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series with a 'Date' datetime column and target value column.
    target_col : str
        Column to model (default: 'Close').
    resampling_freq : str
        Frequency string for resampling (default: 'B' for business days).
    """

    def __init__(self, data: pd.DataFrame, config_path: Path = Path("config/settings.yaml"),
                 target_col: str = 'Close', resampling_freq: str = 'B'):
        self.config = load_config(config_path)
        self.target_col = target_col
        self.resampling_freq = resampling_freq

        self.series = self._extract_series(data, resampling_freq)
        self.model_fit: Optional[SimpleExpSmoothing] = None

    def _extract_series(self, data: pd.DataFrame, resampling_freq: pd.Timedelta) -> pd.Series:
        try:
            series = data[self.target_col]
            # Resample to desired frequency, taking the mean of each period
            resampled_series = series.resample(resampling_freq).mean()
            return resampled_series.dropna()
        except KeyError:
            raise KeyError(f"Could not find '{self.target_col}' in data.")

    def fit(self, smoothing_level: Optional[float] = None):
        self.model_fit = SimpleExpSmoothing(self.series).fit(smoothing_level=smoothing_level)
        print(f"[INFO] SES model fitted with smoothing level: {smoothing_level}")

    def forecast(self, steps: int) -> pd.Series:
        if self.model_fit is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        forecast_values = self.model_fit.forecast(steps)
        forecast_index = pd.date_range(start=self.series.index[-1] + pd.Timedelta(days=1),
                                       periods=steps, freq=self.resampling_freq)
        forecast_values.index = forecast_index
        return forecast_values

    def plot_forecast(self, ax=None, steps: int = 5):
        """
        Plot the historical data and SES forecast.
        """
        if self.model_fit is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        forecast_series = self.forecast(steps)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.series, label="Historical", color="black")
        ax.plot(forecast_series, label="Forecast", color="green")
        ax.axvline(self.series.index[-1], color="gray", linestyle="--", label="Forecast Start")

        ax.set_title(f"SES Forecast for {self.target_col}")
        ax.set_xlabel("Date")
        ax.set_ylabel(self.target_col)
        ax.grid(True)
        ax.legend()

        return ax


# Example usage:
if __name__ == "__main__":
    config_path = Path("config/settings.yaml")

    # Load data using StockDataLoader
    from utils.data_loader import StockDataLoader
    loader = StockDataLoader(config_path)
    df = loader.load_data()

    # Run SES model
    ses_model = SimpleExpSmoothingModel(df, config_path)
    ses_model.fit()
    forecast = ses_model.forecast(steps=5)
    print(forecast)

    # Plot forecast
    ax = ses_model.plot_forecast(steps=5)
    ax.set_xlim(left=19700)
    plt.show()
