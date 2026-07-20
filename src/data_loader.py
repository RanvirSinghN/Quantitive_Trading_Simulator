from pathlib import Path
import sqlite3

import yfinance as yf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "quant_trading.db"


def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data for a given ticker symbol.
    """
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    return data


def initialise_database():
    """
    Creates the SQLite database and prices table if they do not already exist.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                adj_close REAL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (ticker, date)
            )
            """
        )


def save_data_to_database(data, ticker):
    """
    Cleans and saves downloaded OHLCV data to the SQLite prices table.
    """
    initialise_database()
    cleaned_data = clean_data(data)

    rows = cleaned_data.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    rows["ticker"] = ticker.upper()
    if "adj_close" not in rows.columns:
        rows["adj_close"] = None

    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    rows = rows[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (
                ticker, date, open, high, low, close, adj_close, volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows.itertuples(index=False, name=None),
        )

    return DATABASE_PATH


def load_data_from_database(ticker=None):
    """
    Loads price data from the SQLite prices table.
    """
    initialise_database()

    query = "SELECT * FROM prices"
    params = []

    if ticker is not None:
        query += " WHERE ticker = ?"
        params.append(ticker.upper())

    query += " ORDER BY ticker, date"

    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params, parse_dates=["date"])


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

if __name__ == "__main__":
    print("test")
