import pandas as pd
import math
from collections import Counter
from src.backtester import run_simulation, import_selected_tickers, delete_temp_database
from IPython.display import display


if __name__ == "__main__":
    print("Only back testing available at the moment.")
    # User inputs: edit these three values, then run.
    TOTAL_STARTING_CASH = 20_000.00
    TICKERS = ["AAPL", "MSFT", "NVDA"]
    SIMULATION_START_DATE = "2020-03-16"
    SELL_REMAINING_AT_END = True  # Use None to keep the per-stock interactive prompt.

    MINIMUM_SIMULATION_DATE = pd.Timestamp("2020-03-16")
    MARKET_DATA_START_DATE = "2020-01-01"
    MINIMUM_PRICE_HISTORY_ROWS = 50  # Longest feature lookback is the 50-day MA.

    # Validate portfolio-level inputs before downloading or changing database data.
    total_starting_cash = float(TOTAL_STARTING_CASH)
    if not math.isfinite(total_starting_cash) or total_starting_cash <= 0:
        raise ValueError("TOTAL_STARTING_CASH must be a finite number greater than zero.")
    if SELL_REMAINING_AT_END is not None and not isinstance(SELL_REMAINING_AT_END, bool):
        raise ValueError("SELL_REMAINING_AT_END must be True, False, or None.")
    if not isinstance(TICKERS, (list, tuple)) or not TICKERS:
        raise ValueError("TICKERS must be a non-empty list or tuple of ticker symbols.")
    if any(not isinstance(ticker, str) for ticker in TICKERS):
        raise ValueError("Every ticker must be supplied as text.")

    tickers = [ticker.strip().upper() for ticker in TICKERS]
    if any(not ticker for ticker in tickers):
        raise ValueError("Ticker symbols cannot be empty or whitespace only.")
    duplicate_tickers = sorted(
        ticker for ticker, count in Counter(tickers).items() if count > 1
    )
    if duplicate_tickers:
        raise ValueError(f"Duplicate tickers are not allowed: {duplicate_tickers}")

    try:
        simulation_start = pd.to_datetime(
            SIMULATION_START_DATE, format="%Y-%m-%d", errors="raise"
        ).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("SIMULATION_START_DATE must use YYYY-MM-DD format.") from error
    if simulation_start < MINIMUM_SIMULATION_DATE:
        raise ValueError("SIMULATION_START_DATE must be on or after 2020-03-16.")

    # Yahoo's end date is exclusive, so today requests all completed sessions before today.
    market_data_end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    if simulation_start >= pd.Timestamp(market_data_end_date):
        raise ValueError("SIMULATION_START_DATE must be before today.")


    downloaded_market_data, table_status = import_selected_tickers(
        tickers=tickers,
        market_start_date=MARKET_DATA_START_DATE,
        start_date=SIMULATION_START_DATE,
        end_date=market_data_end_date,
    )

    allocated_cash = total_starting_cash / len(tickers)
    simulation_states = {}
    trade_journals = {}
    per_stock_rows = []
    starting_date = simulation_start.strftime("%Y-%m-%d")

    for ticker in tickers:
        state, trade_log, summary = run_simulation(
            ticker=ticker,
            starting_date=starting_date,
            initial_cash=allocated_cash,
            sell_remaining=SELL_REMAINING_AT_END,
        )
        simulation_states[ticker] = state

        # trade_journal.py has no API yet; keep each backtester log separate here.
        trade_journals[ticker] = trade_log.assign(ticker=ticker)

        final_value = float(summary["final_value"])
        profit_loss = final_value - allocated_cash
        per_stock_rows.append(
            {
                "ticker": ticker,
                "allocated_starting_cash": allocated_cash,
                "final_portfolio_value": final_value,
                "profit_loss": profit_loss,
                "return_pct": 100 * profit_loss / allocated_cash,
                "number_of_trades": len(trade_log),
                "remaining_cash": float(summary["final_cash"]),
                "final_shares_held": float(state.shares_held),
            }
        )

    per_stock_summary = pd.DataFrame(per_stock_rows)
    total_final_value = float(per_stock_summary["final_portfolio_value"].sum())
    total_profit_loss = total_final_value - total_starting_cash
    combined_portfolio_summary = pd.DataFrame(
        [
            {
                "total_starting_cash": total_starting_cash,
                "total_final_portfolio_value": total_final_value,
                "total_profit_loss": total_profit_loss,
                "overall_return_pct": 100 * total_profit_loss / total_starting_cash,
                "total_number_of_trades": int(
                    per_stock_summary["number_of_trades"].sum()
                ),
                "total_remaining_cash": float(
                    per_stock_summary["remaining_cash"].sum()
                ),
            }
        ]
    )

    # Reconcile from the per-stock rows; isclose handles normal binary-float noise.
    assert math.isclose(
        per_stock_summary["allocated_starting_cash"].sum(),
        total_starting_cash,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert combined_portfolio_summary.at[0, "total_final_portfolio_value"] == (
        per_stock_summary["final_portfolio_value"].sum()
    )
    assert combined_portfolio_summary.at[0, "total_number_of_trades"] == (
        per_stock_summary["number_of_trades"].sum()
    )

    print("Required database rows by ticker:")
    display(table_status)
    print("Per-stock simulation summary:")
    display(per_stock_summary.round(4))
    print("Combined portfolio summary (reconciled to the per-stock rows):")
    display(combined_portfolio_summary.round(4))

    delete_temp_database(tickers=tickers, downloaded_market_data=downloaded_market_data)
