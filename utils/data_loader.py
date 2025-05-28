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

    def _download_data(self) -> pd.DataFrame:
        return (
            # Download data from Yahoo Finance
            yf.download(self.ticker, start=self.start_date, end=self.end_date)
            # Set ticker as separate column
            .stack(level='Ticker', future_stack=True)
            # Reset index to flatten the DataFrame
            .reset_index()
        )

    def load_data(self) -> pd.DataFrame:
        if self.file_path.exists():
            print(f"[INFO] Loading local file: {self.file_path}")
            # Load the CSV file, ensuring 'Date' is parsed as datetime
            df = pd.read_csv(self.file_path, parse_dates=["Date"])
        else:
            print(f"[INFO] Downloading data for {self.ticker} from Yahoo Finance")
            df = self._download_data()
            if df.empty:
                raise ValueError(f"[ERROR] No data downloaded for ticker '{self.ticker}'.")
            self.store_data(df)

        return df

    def store_data(self, df: pd.DataFrame):
        """Store the DataFrame to a CSV file."""
        df.to_csv(self.file_path, header=True, index=False)
        print(f"[INFO] Saved to {self.file_path}")


# Example usage:
if __name__ == "__main__":
    loader = StockDataLoader("config/settings.yaml")
    df = loader.load_data()
    print(df.head())

