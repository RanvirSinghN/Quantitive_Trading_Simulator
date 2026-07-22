import math
import sqlite3
import pandas as pd

from src.strategy import DATABASE_PATH

def inverse_volatility_distribution(
    tickers: list[str] | tuple[str, ...],
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Return each ticker's inverse-volatility capital allocation percentage.

    The function reads the existing 20-day volatility values from the features
    table and uses the latest date shared by every requested ticker. Supplying
    as_of_date prevents later observations from leaking into a historical test.
    """
    selected_tickers = [ticker.strip().upper() for ticker in tickers]

    placeholders = ", ".join("?" for _ in selected_tickers)
    query = f"""
        SELECT ticker, date, volatility_20
        FROM features
        WHERE ticker IN ({placeholders})
          AND volatility_20 IS NOT NULL
    """
    params: list[str] = selected_tickers.copy()

    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date, errors="raise").normalize()
        query += " AND date <= ?"
        params.append(cutoff.strftime("%Y-%m-%d"))

    query += " ORDER BY date, ticker"
    with sqlite3.connect(DATABASE_PATH) as connection:
        volatility_data = pd.read_sql_query(
            query,
            connection,
            params=params,
            parse_dates=["date"],
        )

    volatility_by_date = volatility_data.pivot(
        index="date", columns="ticker", values="volatility_20"
    ).reindex(columns=selected_tickers)
    common_history = volatility_by_date.dropna(how="any")
    if common_history.empty:
        raise ValueError(
            "The selected tickers have no common date with usable volatility data."
        )

    allocation_date = common_history.index[-1]
    volatility = common_history.loc[allocation_date].astype(float)
    if (volatility <= 0).any() or not volatility.map(math.isfinite).all():
        raise ValueError("Volatility values must be finite and greater than zero.")

    inverse_volatility = 1.0 / volatility
    distribution = (100.0 * inverse_volatility / inverse_volatility.sum()).rename(
        "capital_distribution_pct"
    )
    distribution.index.name = "ticker"
    distribution.attrs["allocation_date"] = allocation_date
    return distribution