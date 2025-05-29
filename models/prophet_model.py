from pathlib import Path
import pandas as pd
from prophet import Prophet
from typing import Optional
import matplotlib.pyplot as plt

from utils.general_utils import load_config
from utils.data_loader import StockDataLoader


class ProphetModel:
    """
    Prophet model wrapper for stock price forecasting.
    """

    def __init__(self, data: pd.DataFrame, config_path: Path = Path("config/settings.yaml"),
                 target_col: str = 'Close', resampling_freq: str = 'B'):
        self.config = load_config(config_path)
        self.target_col = target_col
        self.resampling_freq = resampling_freq

        # Prepare data for Prophet
        self.df_prophet = self._prepare_data(data)

        self.model: Optional[Prophet] = None
        self.forecast_df: Optional[pd.DataFrame] = None

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in data.")

        series = df[self.target_col].resample(self.resampling_freq).mean().dropna()

        # Prophet needs two columns: 'ds' for date, 'y' for value
        prophet_df = series.reset_index()
        prophet_df.columns = ['ds', 'y']
        return prophet_df

    def fit(self):
        self.model = Prophet()
        self.model.fit(self.df_prophet)
        print("[INFO] Prophet model fitted.")

    def forecast(self, steps: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        last_date = self.df_prophet['ds'].max()
        future = self.model.make_future_dataframe(periods=steps, freq=self.resampling_freq)
        self.forecast_df = self.model.predict(future)

        return self.forecast_df.tail(steps)[['ds', 'yhat']]

    def plot_forecast(self, ax=None, steps: int = 5, include_conf_int: bool = True):
        """
        Plot the historical series and Prophet forecast.

        Parameters:
        -----------
        steps : int
            Number of future steps to forecast.
        include_conf_int : bool
            Whether to include confidence intervals.
        """
        if self.model is None or self.forecast_df is None:
            raise RuntimeError("Model not fitted or forecast not generated.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Historical data
        ax.plot(self.df_prophet['ds'], self.df_prophet['y'], label='Historical', color='black')

        # Split forecast into in-sample and out-of-sample
        last_date = self.df_prophet['ds'].max()
        fitted = self.forecast_df[self.forecast_df['ds'] <= last_date]
        future = self.forecast_df[self.forecast_df['ds'] > last_date].head(steps)

        # Fitted values (past): solid blue
        ax.plot(fitted['ds'], fitted['yhat'], label='Fitted', color='blue')

        # Forecast values (future): dashed blue
        ax.plot(future['ds'], future['yhat'], label='Forecast', color='blue', linestyle='--')

        # Confidence interval (future only)
        if include_conf_int and not future.empty:
            ax.fill_between(future['ds'],
                            future['yhat_lower'],
                            future['yhat_upper'],
                            color='blue', alpha=0.2, label='Confidence Interval')

        ax.axvline(last_date, color='gray', linestyle='--', label='Forecast Start')
        ax.set_title(f"Prophet Forecast for {self.target_col}")
        ax.set_xlabel("Date")
        ax.set_ylabel(self.target_col)
        ax.legend()
        ax.grid(True)

        return ax


# Example usage
if __name__ == "__main__":
    # Load data
    config_path = Path("config/settings.yaml")
    loader = StockDataLoader(config_path)
    df = loader.load_data()

    # Prophet model
    model = ProphetModel(df, config_path, target_col='Close', resampling_freq='B')
    model.fit()
    forecast_df = model.forecast(steps=5)
    print(forecast_df)

    # Plot
    ax = model.plot_forecast(steps=5)
    ax.set_xlim(left=19000, right=19750)
    ax.set_ylim(bottom=140)
    plt.tight_layout()
    plt.show()
