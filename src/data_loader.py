from pathlib import Path

import yfinance as yf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_csv"


def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data for a given ticker symbol.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def save_data_to_csv(data, filename):
    """
    Cleans and saves the downloaded data to the data csv folder.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_data = clean_data(data)
    output_path = DATA_DIR / filename
    cleaned_data.to_csv(output_path)
    return output_path


def load_data_from_csv(filename):
    """
    Loads data from the data csv folder.
    """
    input_path = DATA_DIR / filename
    return pd.read_csv(input_path)


def clean_data(data):
    """
    Applies basic textbook cleaning to historical OHLCV market data.
    """
    cleaned = data.copy()

    # Flatten yfinance multi-level columns into simple OHLCV column names.
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = cleaned.columns.get_level_values(0)

    # Move the date/index values into a normal column so they are easy to clean.
    cleaned = cleaned.reset_index()
    if "Date" in cleaned.columns:
        date_column = "Date"
    elif "Price" in cleaned.columns:
        date_column = "Price"
    else:
        date_column = cleaned.columns[0]

    cleaned = cleaned.rename(columns={date_column: "Date"})
    cleaned = cleaned.drop(columns=["index"], errors="ignore")

    # Convert valid date strings to datetime and remove non-data header rows.
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], format="%Y-%m-%d", errors="coerce")
    cleaned = cleaned.dropna(subset=["Date"])

    # Keep one row per trading date and enforce chronological order.
    cleaned = cleaned.drop_duplicates(subset=["Date"], keep="last")
    cleaned = cleaned.sort_values("Date")

    # Confirm the basic market data fields are present before cleaning values.
    price_columns = ["Open", "High", "Low", "Close"]
    required_columns = price_columns + ["Volume"]
    missing_columns = [column for column in required_columns if column not in cleaned.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert prices and volume into numeric values; invalid entries become NaN.
    for column in required_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Drop rows that cannot be safely used for OHLCV analysis.
    cleaned = cleaned.dropna(subset=required_columns)

    # Remove rows that violate basic OHLCV market logic.
    valid_prices = (
        (cleaned["High"] >= cleaned["Low"])
        & (cleaned["High"] >= cleaned["Open"])
        & (cleaned["High"] >= cleaned["Close"])
        & (cleaned["Low"] <= cleaned["Open"])
        & (cleaned["Low"] <= cleaned["Close"])
        & (cleaned["Volume"] >= 0)
    )
    cleaned = cleaned.loc[valid_prices]

    # Use Date as the index for time-series analysis.
    cleaned = cleaned.set_index("Date")
    return cleaned

appl = download_data("AAPL", "2020-01-01", "2023-01-01")
msft = download_data("MSFT", "2020-01-01", "2023-01-01")

save_data_to_csv(appl, "AAPL_data.csv")
save_data_to_csv(msft, "MSFT_data.csv")