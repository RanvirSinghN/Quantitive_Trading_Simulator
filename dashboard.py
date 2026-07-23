"""Simple Streamlit dashboard for the existing historical simulation pipeline."""

from datetime import date, timedelta
import math
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# Make project imports work when Streamlit launches this file from dashboard/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester import (  
    delete_temp_database,
    import_selected_tickers,
    run_simulation,
    create_trade_chart,
)
from src.risk_manager import inverse_volatility_distribution  


# These values match the historical-data assumptions currently used in main.py.
MINIMUM_SIMULATION_DATE = date(2020, 3, 16)
MARKET_DATA_START_DATE = "2020-01-01"
COMMON_TECH_STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
}
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]


def format_ticker_option(ticker: str) -> str:
    """Show a friendly company label for the built-in dropdown choices."""
    normalised_ticker = ticker.strip().upper()
    company = COMMON_TECH_STOCKS.get(normalised_ticker)
    return f"{company} ({normalised_ticker})" if company else normalised_ticker


def build_ticker_list(selected_tickers: list[str]) -> list[str]:
    """Normalise selected and user-entered Yahoo tickers and remove duplicates."""
    tickers = [
        ticker.strip().upper()
        for selection in selected_tickers
        for ticker in str(selection).split(",")
        if ticker.strip()
    ]

    # dict preserves entry order while removing repeated ticker symbols.
    return list(dict.fromkeys(tickers))


def run_dashboard_simulation(
    total_starting_cash: float,
    tickers: list[str],
    simulation_start_date: date,
    sell_remaining_at_end: bool,
) -> dict[str, object]:
    """Run the same import, risk allocation, and backtest flow as main.py."""
    starting_date = simulation_start_date.strftime("%Y-%m-%d")

    # Yahoo's end date is exclusive, so today includes completed sessions before today.
    market_data_end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    downloaded_market_data: dict[str, pd.DataFrame] = {}

    try:
        # Reuse the existing pipeline to download, validate, store, feature-engineer,
        # and create signals for every selected ticker.
        downloaded_market_data, table_status = import_selected_tickers(
            tickers=tickers,
            market_start_date=MARKET_DATA_START_DATE,
            start_date=starting_date,
            end_date=market_data_end_date,
        )

        # Reuse the risk manager to allocate more cash to lower-volatility stocks.
        cash_distribution = inverse_volatility_distribution(
            tickers,
            as_of_date=starting_date,
        )
        allocation_date = cash_distribution.attrs.get("allocation_date")

        per_stock_rows: list[dict[str, object]] = []
        trade_journals: dict[str, pd.DataFrame] = {}
        trade_charts: dict[str, object] = {}

        # Run the existing simulator separately for each risk-weighted allocation.
        for ticker in tickers:
            allocation_percentage = float(cash_distribution[ticker])
            allocated_cash = allocation_percentage * total_starting_cash / 100.0
            state, trade_log, summary = run_simulation(
                ticker=ticker,
                starting_date=starting_date,
                initial_cash=allocated_cash,
                sell_remaining=sell_remaining_at_end,
            )

            trade_journals[ticker] = trade_log
            trade_charts[ticker] = create_trade_chart(
                price_data=downloaded_market_data[ticker],
                trade_log=trade_log,
                ticker=ticker,
                starting_date=starting_date,
            )
            final_value = float(summary["final_value"])
            profit_loss = final_value - allocated_cash
            per_stock_rows.append(
                {
                    "Ticker": ticker,
                    "Allocation (%)": allocation_percentage,
                    "Starting cash (£)": allocated_cash,
                    "Final value (£)": final_value,
                    "Profit / loss (£)": profit_loss,
                    "Return (%)": 100.0 * profit_loss / allocated_cash,
                    "Trades": len(trade_log),
                    "Remaining cash (£)": float(summary["final_cash"]),
                    "Shares held": float(state.shares_held),
                }
            )

        # Reconcile the combined portfolio directly from the per-stock results.
        per_stock_summary = pd.DataFrame(per_stock_rows)
        total_final_value = float(per_stock_summary["Final value (£)"].sum())
        total_profit_loss = total_final_value - total_starting_cash
        overall_return_percentage = 100.0 * total_profit_loss / total_starting_cash
        total_trades = int(per_stock_summary["Trades"].sum())

        if not math.isclose(
            float(per_stock_summary["Starting cash (£)"].sum()),
            total_starting_cash,
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise RuntimeError("Risk allocations do not reconcile to starting cash.")

        return {
            "allocation_date": allocation_date,
            "table_status": table_status,
            "per_stock_summary": per_stock_summary,
            "trade_journals": trade_journals,
            "trade_charts": trade_charts,
            "total_final_value": total_final_value,
            "total_profit_loss": total_profit_loss,
            "overall_return_percentage": overall_return_percentage,
            "total_trades": total_trades,
        }
    finally:
        # The existing main flow treats these database rows as temporary too.
        if downloaded_market_data:
            delete_temp_database(
                tickers=tickers,
                downloaded_market_data=downloaded_market_data,
            )


def display_simulation_results(
    results: dict[str, object],
    starting_cash: float,
) -> None:
    """Render a completed simulation and its optional per-stock trade charts."""
    # Show the combined portfolio result first for quick interpretation.
    st.subheader("Portfolio result")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Starting cash", f"£{starting_cash:,.2f}")
    metric_columns[1].metric(
        "Final value",
        f"£{results['total_final_value']:,.2f}",
    )
    metric_columns[2].metric(
        "Profit / loss",
        f"£{results['total_profit_loss']:,.2f}",
        delta=f"{results['overall_return_percentage']:.2f}%",
    )
    metric_columns[3].metric("Total trades", results["total_trades"])

    # Display risk allocation and outcome for each selected stock.
    st.subheader("Results by stock")
    allocation_date = results["allocation_date"]
    if allocation_date is not None:
        st.caption(
            "Starting cash was allocated using inverse volatility as of "
            f"{pd.Timestamp(allocation_date).date()}."
        )
    st.dataframe(
        results["per_stock_summary"].round(4),
        hide_index=True,
        width="stretch",
    )

    # Keep each journal and its price chart together without cluttering the page.
    st.subheader("Trade journals")
    for ticker, trade_log in results["trade_journals"].items():
        with st.expander(f"{ticker} trades ({len(trade_log)})"):
            if trade_log.empty:
                st.info("No trades were generated for this stock.")
            else:
                display_trade_log = trade_log.copy()
                numeric_columns = display_trade_log.select_dtypes(
                    include="number"
                ).columns
                display_trade_log[numeric_columns] = display_trade_log[
                    numeric_columns
                ].round(4)
                st.dataframe(
                    display_trade_log,
                    hide_index=True,
                    width="stretch",
                )

            if st.button("Show chart", key=f"show_trade_chart_{ticker}"):
                st.caption(
                    "Green triangles are BUY executions; red triangles are SELL "
                    "executions. Markers use execution prices, so they may not sit "
                    "exactly on the daily closing-price line."
                )
                st.plotly_chart(
                    results["trade_charts"][ticker],
                    width="stretch",
                    config={"displaylogo": False},
                )

    # Expose pipeline row counts as a small diagnostic rather than main output.
    with st.expander("Data pipeline details"):
        st.dataframe(
            results["table_status"],
            hide_index=True,
            width="stretch",
        )


# Configure the page and explain that this is historical research, not execution.
st.set_page_config(page_title="Quant Trading Simulator", layout="wide")
st.title("Quant Trading Simulator")
st.caption(
    "Historical backtesting only. This dashboard does not place or recommend trades."
)

# Keep all inputs together and run only when the form is submitted.
with st.form("simulation_inputs"):
    starting_cash = st.number_input(
        "Starting portfolio cash (£)",
        min_value=1.0,
        value=20_000.0,
        step=1_000.0,
        help="This is the cash available to the simulator, not a manually chosen stock price.",
    )
    selected_tickers = st.multiselect(
        "Stocks",
        options=list(COMMON_TECH_STOCKS),
        default=DEFAULT_TICKERS,
        format_func=format_ticker_option,
        accept_new_options=True,
        placeholder="Choose a stock or type a ticker symbol",
        help=(
            "Choose common tech stocks from the dropdown or type any Yahoo Finance "
            "ticker symbol and press Enter."
        ),
    )
    simulation_date = st.date_input(
        "Start trading from",
        value=MINIMUM_SIMULATION_DATE,
        min_value=MINIMUM_SIMULATION_DATE,
        max_value=date.today() - timedelta(days=1),
    )
    sell_remaining = st.checkbox(
        "Sell any remaining shares at the final available closing price",
        value=True,
    )
    submitted = st.form_submit_button("Run simulation", type="primary")


if submitted:
    # Clear older results before attempting a new simulation.
    st.session_state.pop("simulation_results", None)
    st.session_state.pop("simulation_starting_cash", None)

    # Build one ticker list and pass it to the same functions used by main.py.
    tickers = build_ticker_list(selected_tickers)

    if not tickers:
        st.warning("Choose at least one stock before running the simulation.")
    else:
        try:
            with st.spinner("Downloading data and running the simulation..."):
                results = run_dashboard_simulation(
                    total_starting_cash=float(starting_cash),
                    tickers=tickers,
                    simulation_start_date=simulation_date,
                    sell_remaining_at_end=sell_remaining,
                )
        except Exception as error:
            st.error(f"The simulation could not be completed: {error}")
        else:
            # Persist results so a chart-button rerun does not repeat the backtest.
            st.session_state["simulation_results"] = results
            st.session_state["simulation_starting_cash"] = float(starting_cash)
            st.success("Simulation completed.")


# Render saved results on normal reruns and when a Show chart button is clicked.
saved_results = st.session_state.get("simulation_results")
saved_starting_cash = st.session_state.get("simulation_starting_cash")
if saved_results is not None and saved_starting_cash is not None:
    display_simulation_results(saved_results, saved_starting_cash)
