from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml
from utils.general_utils import project_root


class StockDataLoader:
    def __init__(self, config_path=Path("config/settings.yaml")):
        self.config = self._load_config(config_path)
        self.ticker = self.config['ticker']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.data_dir = project_root.joinpath("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / f"{self.ticker}.csv"

    @staticmethod
    def _load_config(path: Path):
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            path = project_root.joinpath(path)

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_data(self) -> pd.DataFrame:
        if self.file_path.exists():
            print(f"[INFO] Loading local file: {self.file_path}")
            # Load the CSV file with multi-index columns
            df = pd.read_csv(self.file_path, header=[0, 1], index_col=0, parse_dates=True)
        else:
            print(f"[INFO] Downloading data for {self.ticker} from Yahoo Finance")
            df = self._download_data()
            df = self._clean_data(df)
            self.store_data(df)  # store cleaned data after loading/downloading
            print(f"[INFO] Saved to {self.file_path}")

        return df

    def _download_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        if df.empty:
            raise ValueError(f"[ERROR] No data downloaded for ticker '{self.ticker}'.")

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"[ERROR] Failed to convert index to datetime: {e}")

        df = df.dropna()
        df = df.sort_index()

        # Try to set frequency to business days for compatibility with time series models
        try:
            df.index.freq = pd.infer_freq(df.index)
            if df.index.freq is None:
                df.index.freq = 'B'  # fallback
        except Exception as e:
            print(f"[WARNING] Could not infer frequency. Reason: {e}")
            df.index.freq = None

        return df

    def store_data(self, df: pd.DataFrame):
        """Save cleaned data to CSV file."""
        df.to_csv(self.file_path)
        print(f"[INFO] Data stored at {self.file_path}")


# Example usage:
if __name__ == "__main__":
    loader = StockDataLoader("config/settings.yaml")
    df = loader.load_data()
    print(df.head())


