import plotly.express as px
import pandas as pd

def create_equity_curve(equity_log: pd.DataFrame) -> px.Figure:
    return px.line(
        equity_log,
        x="date",
        y="equity",
        title="Simulated equity curve",
    )