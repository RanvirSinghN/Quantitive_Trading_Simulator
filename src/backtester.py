from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys
import pandas as pd
import sqlite3
from IPython.display import display
import plotly.graph_objects as go
from datetime import date

CURRENT_DIRECTORY = Path.cwd().resolve()

from src.data_loader import clean_data, download_data, save_data_to_database
from src.feature_engineering import calculate_and_save_features
from src.strategy import DATABASE_PATH, calculate_and_save_signals

@dataclass
class SimulationState:
    ticker: str 
    starting_date: str
    cash: float = 0.0
    shares_held: float = 0.0
    entry_price: Optional[float] = None

    @property
    def has_position(self) -> bool:
        return self.shares_held > 0
    
def load_simulation_data(ticker: str, starting_date: str) -> pd.DataFrame:
    query = """
        SELECT
            s.ticker,
            s.date AS signal_date,
            execution_price.date AS execution_date,
            execution_price.open AS execution_open,
            execution_price.close AS execution_close,
            s.signal_label,
            s.signal_score,
            s.action_if_held,
            s.action_if_not_held,
            s.reasons
        FROM signals AS s
        INNER JOIN prices AS execution_price
            ON execution_price.ticker = s.ticker
            AND execution_price.date = (
                SELECT MIN(next_price.date)
                FROM prices AS next_price
                WHERE next_price.ticker = s.ticker
                  AND next_price.date > s.date
            )
        WHERE s.ticker = ?
          AND s.date >= ?
        ORDER BY s.date
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        simulation_data = pd.read_sql_query(
            query,
            connection,
            params=(ticker.upper(), starting_date),
            parse_dates=["signal_date", "execution_date"],
        )

    if not simulation_data.empty:
        executes_later = (
            simulation_data["execution_date"]
            > simulation_data["signal_date"]
        ).all()
        if not executes_later:
            raise ValueError("Every trade must execute after its signal date.")

    return simulation_data

def run_simulation(
    ticker: str,
    starting_date: str,
    initial_cash: float,
    sell_remaining: Optional[bool] = None,
) -> tuple[SimulationState, pd.DataFrame, dict[str, float]]:
    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero.")
    if sell_remaining is not None and not isinstance(sell_remaining, bool):
        raise ValueError("sell_remaining must be True, False, or None.")

    simulation_data = load_simulation_data(ticker, starting_date)
    if simulation_data.empty:
        raise ValueError("No signals with a following trading day were found.")

    state = SimulationState(
        ticker=ticker.upper(),
        starting_date=starting_date,
        cash=float(initial_cash),
    )
    trades = []

    for row in simulation_data.itertuples(index=False):
        action = (
            row.action_if_held
            if state.has_position
            else row.action_if_not_held
        )
        execution_price = float(row.execution_open)

        if execution_price <= 0:
            raise ValueError(f"Invalid execution price: {execution_price}")

        if action == "BUY":
            if state.has_position:
                raise ValueError("Received BUY while a position was already held.")

            shares_bought = state.cash / execution_price
            state.shares_held = shares_bought
            state.cash = 0.0
            state.entry_price = execution_price
            trade_shares = shares_bought

        elif action == "SELL":
            if not state.has_position:
                raise ValueError("Received SELL while no position was held.")

            trade_shares = state.shares_held
            state.cash += trade_shares * execution_price
            state.shares_held = 0.0
            state.entry_price = None

        elif action in {"HOLD", "HOLD_CASH"}:
            continue

        else:
            raise ValueError(f"Unknown action: {action}")

        trades.append(
            {
                "signal_date": row.signal_date,
                "execution_date": row.execution_date,
                "action": action,
                "shares": trade_shares,
                "execution_price": execution_price,
                "signal_label": row.signal_label,
                "signal_score": row.signal_score,
                "reasons": row.reasons,
            }
        )

    final_close = float(simulation_data.iloc[-1]["execution_close"])
    final_value = state.cash + state.shares_held * final_close
    if state.shares_held > 0:
        print(f"Shares remaining: {state.shares_held} at final close price: {final_close:.2f} with value: {state.shares_held * final_close:.2f}")
        should_sell = sell_remaining
        if should_sell is None:
            sell = input("Would you like to sell your remaining shares at the final close price? (y/n): ")
            should_sell = sell.strip().lower() == "y"
        if should_sell:
            shares_sold = state.shares_held
            state.cash += shares_sold * final_close
            state.shares_held = 0.0
            state.entry_price = None
            trades.append(
                {
                    "signal_date": simulation_data.iloc[-1]["signal_date"],
                    "execution_date": simulation_data.iloc[-1]["execution_date"],
                    "action": "SELL",
                    "shares": shares_sold,
                    "execution_price": final_close,
                    "signal_label": simulation_data.iloc[-1]["signal_label"],
                    "signal_score": simulation_data.iloc[-1]["signal_score"],
                    "reasons": "ended with shares held, user opted to sell at final close price",
                }
            )
    summary = {
        "initial_cash": float(f"{initial_cash:.2f}"),
        "final_cash": float(f"{state.cash:.2f}"),
        "shares_held": state.shares_held,
        "final_value": float(f"{final_value:.2f}"),
        "total_return": float(f"{final_value / initial_cash - 1:.4f}"),
    }

    return state, pd.DataFrame(trades), summary

def delete_ticker_data_from_database(selected_tickers):
    """Deletes selected tickers in child-to-parent order for a fresh run."""
    placeholders = ", ".join("?" for _ in selected_tickers)
    with sqlite3.connect(DATABASE_PATH) as connection:
        for table in ("signals", "features", "prices"):
            connection.execute(
                f"DELETE FROM {table} WHERE ticker IN ({placeholders})",
                selected_tickers,
            )

def import_selected_tickers(tickers: list[str], market_start_date: str, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    # Validate every Yahoo download first so an invalid selection cannot cause a partial run.
    downloaded_market_data = {}
    ticker_errors = {}
    for ticker in tickers:
        try:
            downloaded = download_data(
                ticker, market_start_date, end_date
            )
            cleaned = clean_data(downloaded)
            simulation_start = pd.to_datetime(
                start_date, format="%Y-%m-%d", errors="raise"
            ).normalize()
            history_through_start = cleaned.loc[cleaned.index <= simulation_start]
            requested_period = cleaned.loc[cleaned.index >= simulation_start]
            if cleaned.empty:
                raise ValueError("Yahoo Finance returned no usable OHLCV rows.")
            if len(history_through_start) < 50:
                raise ValueError(
                    f"only {len(history_through_start)} usable rows exist through the "
                    f"start date; at least 50 are required"
                )
            if len(requested_period) < 2:
                raise ValueError(
                    "fewer than two usable rows exist on/after the start date"
                )

            downloaded_market_data[ticker] = downloaded
        except Exception as error:
            ticker_errors[ticker] = str(error)

    if ticker_errors:
        error_lines = ["Ticker validation failed:"] + [
            f"- {ticker}: {message}" for ticker, message in ticker_errors.items()
        ]
        raise ValueError("\n".join(error_lines))

    print(f"Validated Yahoo Finance data for: {', '.join(tickers)}")

    # Remove stale rows first, then reuse the existing database pipeline to build tables.
    delete_ticker_data_from_database(tickers)
    try:
        for ticker, downloaded in downloaded_market_data.items():
            save_data_to_database(downloaded, ticker)
        calculate_and_save_features()
        calculate_and_save_signals()

        # Confirm each selected ticker reached all required database tables.
        table_status_rows = []
        with sqlite3.connect(DATABASE_PATH) as connection:
            for ticker in tickers:
                counts = {}
                for table in ("prices", "features", "signals"):
                    counts[table] = connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE ticker = ?", (ticker,)
                    ).fetchone()[0]
                table_status_rows.append({"ticker": ticker, **counts})

        table_status = pd.DataFrame(table_status_rows)
        if (table_status[["prices", "features", "signals"]] == 0).any().any():
            raise RuntimeError("At least one selected ticker has an empty required table.")

        # The simulator runs from this date through the latest downloaded session.
        starting_date = simulation_start.strftime("%Y-%m-%d")
        for ticker in tickers:
            if load_simulation_data(ticker, starting_date).empty:
                raise ValueError(
                    f"{ticker} has no eligible signal with a following execution day "
                    f"on or after {starting_date}."
                )
    except Exception:
        delete_ticker_data_from_database(tickers)
        downloaded_market_data.clear()
        raise

    return downloaded_market_data, table_status


def delete_temp_database(tickers: list[str], downloaded_market_data: dict[str, pd.DataFrame]):
    """Deletes the temporary database file if it exists."""
    delete_ticker_data_from_database(tickers)
    downloaded_market_data.clear()
    downloaded = None
    cleaned = None

    cleanup_rows = []
    with sqlite3.connect(DATABASE_PATH) as connection:
        for ticker in tickers:
            cleanup_rows.append(
                {
                    "ticker": ticker,
                    **{
                        table: connection.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE ticker = ?",
                            (ticker,),
                        ).fetchone()[0]
                        for table in ("prices", "features", "signals")
                    },
                }
            )
    cleanup_status = pd.DataFrame(cleanup_rows)
    assert (cleanup_status[["prices", "features", "signals"]] == 0).all().all()
    print("Temporary database rows after cleanup:")
    display(cleanup_status)

def create_trade_chart(
    price_data: pd.DataFrame,
    trade_log: pd.DataFrame,
    ticker: str,
    starting_date: str | date | pd.Timestamp,
) -> go.Figure:
    """Return a price chart with markers at completed trade execution points.

    The price line uses the downloaded daily closing price from the simulation
    period. BUY and SELL markers use each trade's actual execution date and
    execution price, which may differ from that day's closing price.
    """
    normalised_ticker = ticker.strip().upper()
    if not normalised_ticker:
        raise ValueError("ticker cannot be empty.")

    # Reuse the project's existing Yahoo-data cleaner so the chart sees the same
    # chronological OHLCV rows that feed the simulation pipeline.
    cleaned_prices = clean_data(price_data)
    start = pd.to_datetime(starting_date, errors="raise").normalize()
    chart_prices = cleaned_prices.loc[cleaned_prices.index >= start].copy()
    if chart_prices.empty:
        raise ValueError(f"No {normalised_ticker} prices exist on or after {start.date()}.")

    # An empty journal is valid: the function still returns the stock-price line.
    plotted_trades = trade_log.copy()
    if not plotted_trades.empty:
        required_columns = {"action", "execution_date", "execution_price"}
        missing_columns = required_columns - set(plotted_trades.columns)
        if missing_columns:
            raise ValueError(
                f"Trade log is missing required columns: {sorted(missing_columns)}"
            )

        plotted_trades["action"] = plotted_trades["action"].astype(str).str.upper()
        plotted_trades["execution_date"] = pd.to_datetime(
            plotted_trades["execution_date"],
            errors="raise",
        )
        plotted_trades["execution_price"] = pd.to_numeric(
            plotted_trades["execution_price"],
            errors="raise",
        )

    # Create the interactive closing-price line and show exact values on hover.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_prices.index,
            y=chart_prices["Close"],
            mode="lines",
            name="Closing price",
            line={"color": "#2563eb", "width": 2},
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Closing price: $%{y:,.2f}<extra></extra>"
            ),
        )
    )

    # Split completed trades by action; an empty journal produces empty markers.
    if plotted_trades.empty:
        buys = plotted_trades
        sells = plotted_trades
    else:
        buys = plotted_trades.loc[plotted_trades["action"] == "BUY"]
        sells = plotted_trades.loc[plotted_trades["action"] == "SELL"]

    # Plot green upward triangles at actual BUY executions.
    if not buys.empty:
        figure.add_trace(
            go.Scatter(
                x=buys["execution_date"],
                y=buys["execution_price"],
                mode="markers",
                name="BUY",
                marker={
                    "color": "#16a34a",
                    "line": {"color": "white", "width": 1},
                    "size": 13,
                    "symbol": "triangle-up",
                },
                hovertemplate=(
                    "BUY<br>Date: %{x|%Y-%m-%d}<br>"
                    "Execution price: $%{y:,.2f}<extra></extra>"
                ),
            )
        )

    # Plot red downward triangles at actual SELL executions.
    if not sells.empty:
        figure.add_trace(
            go.Scatter(
                x=sells["execution_date"],
                y=sells["execution_price"],
                mode="markers",
                name="SELL",
                marker={
                    "color": "#dc2626",
                    "line": {"color": "white", "width": 1},
                    "size": 13,
                    "symbol": "triangle-down",
                },
                hovertemplate=(
                    "SELL<br>Date: %{x|%Y-%m-%d}<br>"
                    "Execution price: $%{y:,.2f}<extra></extra>"
                ),
            )
        )

    # Apply a clean financial-chart layout while preserving Plotly zoom and pan.
    figure.update_layout(
        title=f"{normalised_ticker} price and simulated trades",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="closest",
        height=520,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.08,
            "xanchor": "left",
            "yanchor": "bottom",
        },
        margin={"l": 20, "r": 20, "t": 85, "b": 20},
    )

    return figure



