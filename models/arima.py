from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional


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

    def __init__(self, data: pd.DataFrame, ticker: str, p: int = 1, d: int = 0, q: int = 0, target_col: str = 'Close'):
        self.ticker = ticker
        self.p = p
        self.d = d
        self.q = q
        self.target_col = target_col

        # Extract univariate series from multiindex dataframe
        self.series = self._extract_series(data)

        self.model_fit: Optional[ARIMA] = None

    def _extract_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Extracts the target time series for the ticker from the multiindex dataframe.

        Parameters:
        -----------
        data : pd.DataFrame
            MultiIndex columns dataframe (Price, Ticker).

        Returns:
        --------
        pd.Series
            Time series for the specified target_col and ticker.
        """
        try:
            series = data[self.target_col][self.ticker]
        except KeyError:
            raise KeyError(f"Could not find '{self.target_col}' for ticker '{self.ticker}' in data.")

        return series.dropna()

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


# Example usage:
if __name__ == "__main__":
    # # Example DataFrame with MultiIndex columns
    # data = pd.DataFrame({
    #     ('Close', 'AAPL'): [150, 152, 153, 155, 157],
    #     ('Close', 'GOOGL'): [2800, 2820, 2815, 2830, 2840]
    # }, index=pd.date_range(start='2023-01-01', periods=5, freq='B'))
    #
    # model = ARIMAModel(data, ticker='AAPL', p=1, d=0, q=0)
    # model.fit()
    # print(model.summary())
    # forecast = model.forecast(steps=5)
    # print(forecast)

    # Load data using StockDataLoader
    from utils.data_loader import StockDataLoader
    loader = StockDataLoader("config/settings.yaml")
    df = loader.load_data()

    # Assume df is your loaded multiindex dataframe, e.g., from StockDataLoader.load_data()
    arima_model = ARIMAModel(df, ticker='AAPL', p=5, d=1, q=0)
    arima_model.fit()
    arima_model.summary()
    forecast = arima_model.forecast(steps=5)
    print(forecast)