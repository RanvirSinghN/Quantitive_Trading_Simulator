import pandas as pd
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "quant_trading.db"
FEATURE_COLUMNS = [
    "return_1d",
    "ma_20",
    "ma_50",
    "volatility_20",
    "momentum_10",
    "momentum_20",
    "rsi_14",
    "drawdown",
]


def initialise_database():
    """
    Creates the features table in the existing quant trading database.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                return_1d REAL,
                ma_20 REAL,
                ma_50 REAL,
                volatility_20 REAL,
                momentum_10 REAL,
                momentum_20 REAL,
                rsi_14 REAL,
                drawdown REAL,
                PRIMARY KEY (ticker, date),
                FOREIGN KEY (ticker, date) REFERENCES prices (ticker, date)
            )
            """
        )


def _validate_columns(df, required_columns):
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def _sort_prices(df):
    _validate_columns(df, {"ticker", "date"})

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_return_1d(df):
    """
    Adds one-day returns using adjusted close, calculated separately by ticker.
    """
    _validate_columns(df, {"ticker", "date", "adj_close"})

    result = _sort_prices(df)
    result["return_1d"] = result.groupby("ticker")["adj_close"].pct_change()
    return result


def add_moving_averages(df, window):
    """
    Adds a moving average feature without rolling across different tickers.
    """
    if window not in (20, 50):
        raise ValueError("window must be either 20 or 50")

    _validate_columns(df, {"ticker", "date", "adj_close"})

    result = _sort_prices(df)
    result[f"ma_{window}"] = (
        result.groupby("ticker")["adj_close"]
        .transform(lambda prices: prices.rolling(window=window, min_periods=window).mean())
    )

    return result


def add_volatility_20(df):
    """
    Adds 20-day rolling volatility using daily adjusted-close returns.
    """
    _validate_columns(df, {"ticker", "date", "adj_close"})

    result = df.copy()
    if "return_1d" not in result.columns:
        result = add_return_1d(result)
    else:
        result = _sort_prices(result)

    result["volatility_20"] = (
        result.groupby("ticker")["return_1d"]
        .transform(lambda returns: returns.rolling(window=20, min_periods=20).std())
    )
    return result


def add_momentum(df, window):
    """
    Adds momentum over either 10 or 20 trading days.
    """
    if window not in (10, 20):
        raise ValueError("window must be either 10 or 20")

    _validate_columns(df, {"ticker", "date", "adj_close"})

    result = _sort_prices(df)
    result[f"momentum_{window}"] = (
        result.groupby("ticker")["adj_close"]
        .transform(lambda prices: prices / prices.shift(window) - 1)
    )
    return result


def add_rsi_14(df):
    """
    Adds 14-day RSI using adjusted close, calculated separately by ticker.
    """
    _validate_columns(df, {"ticker", "date", "adj_close"})

    result = _sort_prices(df)
    price_change = result.groupby("ticker")["adj_close"].diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.groupby(result["ticker"]).transform(
        lambda values: values.rolling(window=14, min_periods=14).mean()
    )
    average_loss = losses.groupby(result["ticker"]).transform(
        lambda values: values.rolling(window=14, min_periods=14).mean()
    )

    relative_strength = average_gain / average_loss
    result["rsi_14"] = 100 - (100 / (1 + relative_strength))
    result.loc[average_loss == 0, "rsi_14"] = 100
    return result


def add_drawdown(df):
    """
    Adds drawdown from the ticker's historical adjusted-close peak.
    """
    _validate_columns(df, {"ticker", "date", "adj_close"})

    result = _sort_prices(df)
    peak_price = result.groupby("ticker")["adj_close"].cummax()
    result["drawdown"] = result["adj_close"] / peak_price - 1
    return result


def build_features(df):
    """
    Builds the full feature set from a prices DataFrame.
    """
    result = add_return_1d(df)
    result = add_moving_averages(result, 20)
    result = add_moving_averages(result, 50)
    result = add_volatility_20(result)
    result = add_momentum(result, 10)
    result = add_momentum(result, 20)
    result = add_rsi_14(result)
    result = add_drawdown(result)
    return result[["ticker", "date"] + FEATURE_COLUMNS]


def load_prices_from_database():
    """
    Loads prices from SQLite for feature engineering.
    """
    initialise_database()

    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM prices ORDER BY ticker, date",
            conn,
            parse_dates=["date"],
        )


def save_features_to_database(features):
    """
    Saves calculated features to the SQLite features table.
    """
    initialise_database()
    _validate_columns(features, {"ticker", "date", *FEATURE_COLUMNS})

    rows = features.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    rows = rows[["ticker", "date"] + FEATURE_COLUMNS]
    rows = rows.where(pd.notna(rows), None)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO features (
                ticker, date, return_1d, ma_20, ma_50, volatility_20,
                momentum_10, momentum_20, rsi_14, drawdown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows.itertuples(index=False, name=None),
        )

    return DATABASE_PATH


def calculate_and_save_features():
    """
    Calculates all features from prices and saves them into SQLite.
    """
    prices = load_prices_from_database()
    features = build_features(prices)
    save_features_to_database(features)
    return features
