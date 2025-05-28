from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml
from utils.general_utils import project_root, load_config


class StockDataLoader:
    def __init__(self, config_path=Path("config/settings.yaml")):
        self.config = load_config(config_path)
        self.ticker = self.config['ticker']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.data_dir = project_root.joinpath("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / f"{self.ticker}.csv"

    @staticmethod
    def load_config(path: Path):
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            path = project_root.joinpath(path)

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _download_data(self) -> pd.DataFrame:
        return (
            # Download data from Yahoo Finance
            yf.download(self.ticker, start=self.start_date, end=self.end_date)
            # Resample to ensure daily frequency
            .resample('D').ffill()
            # Set ticker as separate column
            .stack(level='Ticker', future_stack=True)
        )

    def load_data(self) -> pd.DataFrame:
        if self.file_path.exists():
            print(f"[INFO] Loading local file: {self.file_path}")
            # Load the CSV file, ensuring 'Date' is parsed as datetime and set as index
            df = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date")
        else:
            print(f"[INFO] Downloading data for {self.ticker} from Yahoo Finance")
            df = self._download_data()
            if df.empty:
                raise ValueError(f"[ERROR] No data downloaded for ticker '{self.ticker}'.")
            self.store_data(df)

        return df

    def store_data(self, df: pd.DataFrame):
        """Store the DataFrame to a CSV file."""
        df.to_csv(self.file_path, header=True, index='Date')
        print(f"[INFO] Saved to {self.file_path}")


# Example usage:
if __name__ == "__main__":
    # Settings
    config_path = Path("config/settings.yaml")

    loader = StockDataLoader(config_path)
    df = loader.load_data()
    print(df.head())

