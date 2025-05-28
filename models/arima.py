import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional


from utils.general_utils import load_config


class ARIMAModel:
    """
    Simple ARIMA model wrapper for stock price forecasting.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with MultiIndex columns (Price, Ticker).
    ticker : str
        The ticker symbol to extract from data.
    p, d, q : int
        ARIMA hyperparameters.
    target_col : str
        Column to model (default: 'Close').

    Attributes:
    -----------
    model_fit : ARIMAResults
        The fitted ARIMA model.
    """

    def __init__(self, data: pd.DataFrame, config_path: Path = Path("config/settings.yaml"),
                 p: int = 1, d: int = 0, q: int = 0,
                 target_col: str = 'Close', resampling_freq='B'):
        self.config = load_config(config_path)
        # self.ticker = self.config['ticker']
        self.p = p
        self.d = d
        self.q = q
        self.target_col = target_col
        self.resampling_freq = resampling_freq

        # Extract univariate series from multiindex dataframe
        self.series = self._extract_series(data, resampling_freq)

        self.model_fit: Optional[ARIMA] = None

    def _extract_series(self, data: pd.DataFrame, resampling_freq: pd.Timedelta) -> pd.Series:
        """
        Extracts the target time series for the ticker from the dataframe.

        Parameters:
        -----------
        data : pd.DataFrame

        Returns:
        --------
        pd.Series
            Time series for the specified target_col, indexed by Date in
            datetime format.
        """
        try:
            series = data[self.target_col]
            # Resample to desired frequency, taking the mean of each period
            resampled_series = series.resample(resampling_freq).mean()
            return resampled_series.dropna()
        except KeyError:
            raise KeyError(f"Could not find '{self.target_col}' in data.")

    def fit(self):
        """
        Fits the ARIMA model to the extracted series.
        """
        self.model_fit = ARIMA(self.series, order=(self.p, self.d, self.q)).fit()
        print(f"[INFO] ARIMA model fitted with order=({self.p},{self.d},{self.q})")

    def forecast(self, steps: int) -> pd.Series:
        """
        Forecast future values.

        Parameters:
        -----------
        steps : int
            Number of future time steps to forecast.

        Returns:
        --------
        pd.Series
            Forecasted values with datetime index.
        """
        if self.model_fit is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        forecast_result = self.model_fit.get_forecast(steps=steps)
        forecast_series = forecast_result.predicted_mean
        forecast_series.index = pd.date_range(start=self.series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='B')

        return forecast_series

    def summary(self):
        """
        Prints the model summary.
        """
        if self.model_fit is None:
            print("[WARN] Model not fitted yet.")
        else:
            print(self.model_fit.summary())

    def plot_forecast(self, ax=None, steps: int = 5, include_conf_int: bool = True):
        """
        Plot the historical series and ARIMA forecast.

        Parameters:
        -----------
        steps : int
            Number of future steps to forecast.
        include_conf_int : bool
            Whether to include confidence intervals.
        """
        if self.model_fit is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        forecast_result = self.model_fit.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        forecast_index = pd.date_range(start=self.series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='B')

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.series, label='Historical', color='black')
        ax.plot(forecast_index, forecast_mean, label='Forecast', color='blue')

        if include_conf_int:
            conf_int = forecast_result.conf_int()
            lower, upper = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
            plt.fill_between(forecast_index, lower, upper, color='blue', alpha=0.2, label='Confidence Interval')

        ax.axvline(self.series.index[-1], color='gray', linestyle='--', label='Forecast Start')
        ax.set_title(f"ARIMA Forecast for {self.target_col}")
        ax.set_xlabel("Date")
        ax.set_ylabel(self.target_col)
        ax.grid(True)

        return ax


# Example usage:
if __name__ == "__main__":
    # Settings
    config_path = Path("config/settings.yaml")

    # Load data using StockDataLoader
    from utils.data_loader import StockDataLoader
    loader = StockDataLoader(config_path)
    df = loader.load_data()

    # Run ARIMA model
    arima_model = ARIMAModel(df, config_path, p=5, d=1, q=0)
    arima_model.fit()
    arima_model.summary()
    forecast = arima_model.forecast(steps=5)

    print(forecast)

    ax = arima_model.plot_forecast(steps=5)


