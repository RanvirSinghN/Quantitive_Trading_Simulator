import plotly.express as px
import pandas as pd

def create_equity_curve(equity_log: pd.DataFrame) -> px.Figure:
    return px.line(
        equity_log,
        x="date",
        y="equity",
        title="Simulated equity curve",
    )

def calc_total_return(equity_log: pd.DataFrame) -> float:
    if equity_log.empty:
        return 0.0
    starting_equity = equity_log.iloc[0]["equity"]
    ending_equity = equity_log.iloc[-1]["equity"]
    return (ending_equity - starting_equity) / starting_equity * 100

def calc_annualized_return(equity_log: pd.DataFrame) -> float:
    if equity_log.empty:
        return 0.0
    starting_equity = equity_log.iloc[0]["equity"]
    ending_equity = equity_log.iloc[-1]["equity"]
    num_days = (equity_log.iloc[-1]["date"] - equity_log.iloc[0]["date"]).days
    if num_days == 0:
        return 0.0
    annualized_return = ((ending_equity / starting_equity) ** (365 / num_days)) - 1
    return annualized_return * 100
