import numpy as np
import pandas as pd
import plotly.express as px


TRADING_DAYS_PER_YEAR = 252


def create_equity_curve(equity_log: pd.DataFrame) -> px.Figure:
    series_column = "series" if "series" in equity_log.columns else None
    return px.line(
        equity_log,
        x="date",
        y="equity",
        color=series_column,
        title=(
            "Strategy vs buy and hold"
            if series_column is not None
            else "Simulated equity curve"
        ),
        labels={"equity": "Portfolio value", "series": "Portfolio"},
    )


def _prepare_equity_log(equity_log: pd.DataFrame) -> pd.DataFrame:
    """Return valid equity observations sorted in chronological order."""
    if equity_log.empty:
        return pd.DataFrame(columns=["date", "equity"])
    if not {"date", "equity"}.issubset(equity_log.columns):
        raise ValueError("equity_log must contain 'date' and 'equity' columns.")

    prepared = equity_log.loc[:, ["date", "equity"]].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["equity"] = pd.to_numeric(prepared["equity"], errors="coerce")
    prepared = (
        prepared.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return prepared


def _daily_returns(equity_log: pd.DataFrame) -> pd.Series:
    prepared = _prepare_equity_log(equity_log)
    returns = prepared["equity"].pct_change(fill_method=None)
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def calc_total_return(
    equity_log: pd.DataFrame,
    starting_equity: float | None = None,
) -> float:
    """Return the strategy's total percentage gain or loss."""
    prepared = _prepare_equity_log(equity_log)
    if prepared.empty:
        return 0.0

    start = float(prepared.iloc[0]["equity"] if starting_equity is None else starting_equity)
    end = float(prepared.iloc[-1]["equity"])
    if start <= 0:
        raise ValueError("starting equity must be greater than zero.")
    return (end / start - 1) * 100


def calc_annualized_return(
    equity_log: pd.DataFrame,
    starting_equity: float | None = None,
) -> float:
    """Return CAGR as a percentage, using elapsed calendar days."""
    prepared = _prepare_equity_log(equity_log)
    if len(prepared) < 2:
        return 0.0

    start = float(prepared.iloc[0]["equity"] if starting_equity is None else starting_equity)
    end = float(prepared.iloc[-1]["equity"])
    elapsed_days = (prepared.iloc[-1]["date"] - prepared.iloc[0]["date"]).days
    if start <= 0:
        raise ValueError("starting equity must be greater than zero.")
    if end < 0 or elapsed_days <= 0:
        return float("nan")

    return ((end / start) ** (365.0 / elapsed_days) - 1) * 100


def calc_max_drawdown(equity_log: pd.DataFrame) -> float:
    """Return the largest peak-to-trough fall as a positive percentage."""
    prepared = _prepare_equity_log(equity_log)
    if prepared.empty:
        return 0.0

    running_peak = prepared["equity"].cummax()
    drawdowns = prepared["equity"].div(running_peak).sub(1)
    return abs(float(drawdowns.min())) * 100


def calc_annualized_volatility(equity_log: pd.DataFrame) -> float:
    """Return daily-return volatility annualized over 252 trading days."""
    returns = _daily_returns(equity_log)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def calc_sharpe_ratio(
    equity_log: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    """Return annualized Sharpe; risk_free_rate is an annual decimal rate."""
    returns = _daily_returns(equity_log)
    if len(returns) < 2:
        return 0.0

    daily_risk_free_rate = risk_free_rate / TRADING_DAYS_PER_YEAR
    volatility = returns.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return float("nan")
    return float(
        (returns.mean() - daily_risk_free_rate)
        / volatility
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def calc_calmar_ratio(
    equity_log: pd.DataFrame,
    starting_equity: float | None = None,
) -> float:
    """Return CAGR divided by maximum drawdown."""
    max_drawdown = calc_max_drawdown(equity_log)
    if max_drawdown == 0:
        return float("nan")
    return calc_annualized_return(equity_log, starting_equity) / max_drawdown


def calc_best_day(equity_log: pd.DataFrame) -> float:
    """Return the largest one-observation percentage gain."""
    returns = _daily_returns(equity_log)
    return float(returns.max() * 100) if not returns.empty else 0.0


def calc_worst_day(equity_log: pd.DataFrame) -> float:
    """Return the largest one-observation percentage loss."""
    returns = _daily_returns(equity_log)
    return float(returns.min() * 100) if not returns.empty else 0.0


def calc_positive_day_rate(equity_log: pd.DataFrame) -> float:
    """Return the percentage of observed return periods above zero."""
    returns = _daily_returns(equity_log)
    return float(returns.gt(0).mean() * 100) if not returns.empty else 0.0


def calc_time_in_market(
    equity_log: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> float:
    """Return the percentage of observed sessions ending with a position held.

    This expects one ticker sleeve that starts in cash. A BUY opens the position
    and a SELL closes it. BUY sessions count as invested; SELL sessions do not.
    """
    prepared_equity = _prepare_equity_log(equity_log)
    if prepared_equity.empty or trade_log.empty:
        return 0.0
    if not {"execution_date", "action"}.issubset(trade_log.columns):
        raise ValueError("trade_log must contain 'execution_date' and 'action' columns.")

    trades = trade_log.loc[:, ["execution_date", "action"]].copy()
    trades["execution_date"] = pd.to_datetime(trades["execution_date"], errors="coerce")
    trades["action"] = trades["action"].astype(str).str.upper()
    trades = trades.dropna(subset=["execution_date"]).sort_values("execution_date")

    position_is_open = False
    invested_sessions = 0
    trade_index = 0
    trade_rows = list(trades.itertuples(index=False))

    for observation_date in prepared_equity["date"]:
        while (
            trade_index < len(trade_rows)
            and trade_rows[trade_index].execution_date <= observation_date
        ):
            action = trade_rows[trade_index].action
            if action == "BUY":
                position_is_open = True
            elif action == "SELL":
                position_is_open = False
            trade_index += 1
        invested_sessions += int(position_is_open)

    return invested_sessions / len(prepared_equity) * 100


def calc_trade_count(trade_log: pd.DataFrame) -> int:
    """Return the number of executed BUY and SELL rows."""
    if trade_log.empty:
        return 0
    if "action" not in trade_log.columns:
        raise ValueError("trade_log must contain an 'action' column.")
    return int(trade_log["action"].astype(str).str.upper().isin(["BUY", "SELL"]).sum())


def calculate_performance_stats(
    equity_log: pd.DataFrame,
    trade_log: pd.DataFrame | None = None,
    starting_equity: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | None]:
    """Return the core equity-curve statistics in one dashboard-friendly dict."""
    return {
        "total_return_pct": calc_total_return(equity_log, starting_equity),
        "annualized_return_pct": calc_annualized_return(equity_log, starting_equity),
        "max_drawdown_pct": calc_max_drawdown(equity_log),
        "annualized_volatility_pct": calc_annualized_volatility(equity_log),
        "sharpe_ratio": calc_sharpe_ratio(equity_log, risk_free_rate),
        "calmar_ratio": calc_calmar_ratio(equity_log, starting_equity),
        "best_day_pct": calc_best_day(equity_log),
        "worst_day_pct": calc_worst_day(equity_log),
        "positive_day_rate_pct": calc_positive_day_rate(equity_log),
        "time_in_market_pct": (
            calc_time_in_market(equity_log, trade_log)
            if trade_log is not None
            else None
        ),
        "trade_count": calc_trade_count(trade_log) if trade_log is not None else None,
    }
