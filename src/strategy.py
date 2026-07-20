from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


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

SIGNAL_COLUMNS = [
    "ticker",
    "date",
    "trend_score",
    "momentum_score",
    "rsi_score",
    "volatility_score",
    "drawdown_score",
    "signal_score",
    "volatility_percentile",
    "raw_signal_label",
    "signal_label",
    "buy_blocked",
    "action_if_held",
    "action_if_not_held",
    "reasons",
]

VOLATILITY_LOOKBACK = 252
VOLATILITY_MIN_HISTORY = 20


def _validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raises a clear error when a required input column is missing."""
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def initialise_signals_table() -> None:
    """Recreates the derived signals table without changing price or feature data."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS signals")
        conn.execute(
            """
            CREATE TABLE signals (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                trend_score INTEGER NOT NULL,
                momentum_score INTEGER NOT NULL,
                rsi_score INTEGER NOT NULL,
                volatility_score INTEGER NOT NULL,
                drawdown_score INTEGER NOT NULL,
                signal_score INTEGER NOT NULL,
                volatility_percentile REAL NOT NULL,
                raw_signal_label TEXT NOT NULL,
                signal_label TEXT NOT NULL,
                buy_blocked INTEGER NOT NULL CHECK (buy_blocked IN (0, 1)),
                action_if_held TEXT NOT NULL,
                action_if_not_held TEXT NOT NULL,
                reasons TEXT NOT NULL,
                PRIMARY KEY (ticker, date),
                FOREIGN KEY (ticker, date) REFERENCES features (ticker, date)
            )
            """
        )


def load_signal_inputs() -> pd.DataFrame:
    """Loads prices and their matching features in chronological order."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                p.ticker,
                p.date,
                p.adj_close,
                p.volume,
                f.return_1d,
                f.ma_20,
                f.ma_50,
                f.volatility_20,
                f.momentum_10,
                f.momentum_20,
                f.rsi_14,
                f.drawdown
            FROM prices AS p
            INNER JOIN features AS f
                ON p.ticker = f.ticker AND p.date = f.date
            ORDER BY p.ticker, p.date
            """,
            conn,
            parse_dates=["date"],
        )


def _past_volatility_percentiles(
    volatility: pd.Series,
    lookback: int = VOLATILITY_LOOKBACK,
    min_history: int = VOLATILITY_MIN_HISTORY,
) -> pd.Series:
    """
    Ranks each observation against prior volatility values only.

    Excluding the current and future observations prevents look-ahead bias. A
    percentile of 0.90 means current volatility is at least as high as 90% of
    the available prior values inside the lookback window.
    """
    values = volatility.to_numpy(dtype=float)
    percentiles = np.full(len(values), np.nan, dtype=float)

    for position, current_value in enumerate(values):
        if np.isnan(current_value):
            continue

        window_start = max(0, position - lookback)
        history = values[window_start:position]
        history = history[~np.isnan(history)]

        if len(history) >= min_history:
            percentiles[position] = float(np.mean(history <= current_value))

    return pd.Series(percentiles, index=volatility.index, dtype=float)


def add_volatility_percentile(inputs: pd.DataFrame) -> pd.DataFrame:
    """Adds a past-only rolling volatility percentile separately per ticker."""
    _validate_columns(inputs, {"ticker", "date", "volatility_20"})

    result = inputs.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    result["volatility_percentile"] = (
        result.groupby("ticker", group_keys=False)["volatility_20"]
        .apply(_past_volatility_percentiles)
        .reset_index(level=0, drop=True)
    )
    return result


def _trend_factor(row: pd.Series) -> tuple[int, str]:
    score = 0
    reasons = []

    if row["adj_close"] > row["ma_20"]:
        score += 10
        reasons.append("price above MA20")
    elif row["adj_close"] < row["ma_20"]:
        score -= 10
        reasons.append("price below MA20")
    else:
        reasons.append("price equal to MA20")

    if row["ma_20"] > row["ma_50"]:
        score += 20
        reasons.append("MA20 above MA50")
    elif row["ma_20"] < row["ma_50"]:
        score -= 20
        reasons.append("MA20 below MA50")
    else:
        reasons.append("MA20 equal to MA50")

    return score, f"Trend {score:+d}: " + ", ".join(reasons)


def _momentum_factor(row: pd.Series) -> tuple[int, str]:
    score = 0
    reasons = []

    if row["momentum_10"] > 0:
        score += 10
        reasons.append("10-day momentum positive")
    elif row["momentum_10"] < 0:
        score -= 10
        reasons.append("10-day momentum negative")
    else:
        reasons.append("10-day momentum flat")

    if row["momentum_20"] > 0:
        score += 15
        reasons.append("20-day momentum positive")
    elif row["momentum_20"] < 0:
        score -= 15
        reasons.append("20-day momentum negative")
    else:
        reasons.append("20-day momentum flat")

    if row["momentum_10"] > 0 and row["momentum_20"] > 0:
        score += 5
        reasons.append("both horizons confirm bullish momentum")
    elif row["momentum_10"] < 0 and row["momentum_20"] < 0:
        score -= 5
        reasons.append("both horizons confirm bearish momentum")
    else:
        reasons.append("momentum horizons do not fully agree")

    return score, f"Momentum {score:+d}: " + ", ".join(reasons)


def _rsi_factor(row: pd.Series) -> tuple[int, str]:
    rsi = float(row["rsi_14"])

    if 50 <= rsi <= 70:
        score = 15
        description = "bullish without being extremely overbought"
    elif 70 < rsi <= 75:
        score = 8
        description = "bullish but becoming stretched"
    elif rsi > 75:
        score = -5
        description = "extremely overbought; chasing risk is elevated"
    elif 40 <= rsi < 50:
        score = -5
        description = "mildly bearish"
    elif 30 <= rsi < 40:
        score = -10
        description = "bearish"
    else:
        score = -15
        description = "oversold, treated as weakness rather than an automatic buy"

    return score, f"RSI {score:+d}: {rsi:.1f}, {description}"


def _volatility_factor(row: pd.Series) -> tuple[int, str]:
    percentile = float(row["volatility_percentile"])

    if percentile <= 0.50:
        score = 15
        description = "normal-to-low risk"
    elif percentile <= 0.80:
        score = 5
        description = "moderately elevated risk"
    elif percentile <= 0.90:
        score = -5
        description = "high risk"
    else:
        score = -15
        description = "extreme risk"

    return (
        score,
        f"Volatility {score:+d}: {percentile:.0%} percentile versus prior history, "
        f"{description}",
    )


def _drawdown_factor(row: pd.Series) -> tuple[int, str]:
    drawdown = float(row["drawdown"])

    if drawdown >= -0.10:
        score = 10
        description = "within 10% of its historical peak"
    elif drawdown >= -0.20:
        score = 5
        description = "moderate drawdown"
    elif drawdown >= -0.30:
        score = -5
        description = "deep drawdown"
    else:
        score = -10
        description = "severe drawdown"

    return score, f"Drawdown {score:+d}: {drawdown:.1%}, {description}"


def classify_signal(score: int) -> str:
    """Maps the continuous factor score to a readable market outlook."""
    if score >= 70:
        return "STRONG_BULLISH"
    if score >= 50:
        return "BULLISH"
    if score <= -70:
        return "STRONG_BEARISH"
    if score <= -50:
        return "BEARISH"
    return "NEUTRAL"


def _buy_guardrails(row: pd.Series) -> list[str]:
    """Returns conditions that prohibit a new buy without forcing a sale."""
    blocks = []

    if row["volatility_percentile"] > 0.90:
        blocks.append("volatility above its prior-history 90th percentile")
    if row["drawdown"] < -0.30:
        blocks.append("drawdown worse than 30%")
    if row["return_1d"] < (-2 * row["volatility_20"]):
        blocks.append("one-day loss greater than two daily volatility units")

    return blocks


def _actions_for_signal(signal_label: str) -> tuple[str, str]:
    """Returns actions for an existing long position and for no position."""
    if signal_label in {"STRONG_BULLISH", "BULLISH"}:
        return "HOLD", "BUY"
    if signal_label in {"STRONG_BEARISH", "BEARISH"}:
        return "SELL", "HOLD_CASH"
    return "HOLD", "HOLD_CASH"


def score_signal_row(row: pd.Series) -> dict[str, object]:
    """Scores one eligible price/feature observation and explains the result."""
    trend_score, trend_reason = _trend_factor(row)
    momentum_score, momentum_reason = _momentum_factor(row)
    rsi_score, rsi_reason = _rsi_factor(row)
    volatility_score, volatility_reason = _volatility_factor(row)
    drawdown_score, drawdown_reason = _drawdown_factor(row)

    signal_score = int(
        trend_score
        + momentum_score
        + rsi_score
        + volatility_score
        + drawdown_score
    )
    raw_signal_label = classify_signal(signal_score)
    block_reasons = _buy_guardrails(row)
    buy_blocked = bool(block_reasons)

    # Guardrails block a proposed new long, but do not manufacture a sell signal.
    if raw_signal_label in {"STRONG_BULLISH", "BULLISH"} and buy_blocked:
        signal_label = "NEUTRAL"
        guardrail_reason = "Buy blocked: " + ", ".join(block_reasons)
    else:
        signal_label = raw_signal_label
        guardrail_reason = (
            "Buy guardrails active: " + ", ".join(block_reasons)
            if buy_blocked
            else "Buy guardrails clear"
        )

    action_if_held, action_if_not_held = _actions_for_signal(signal_label)
    return_reason = (
        f"One-day return {row['return_1d']:+.2%}; "
        f"two-volatility downside threshold {-2 * row['volatility_20']:.2%}"
    )

    return {
        "ticker": row["ticker"],
        "date": row["date"],
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "rsi_score": rsi_score,
        "volatility_score": volatility_score,
        "drawdown_score": drawdown_score,
        "signal_score": signal_score,
        "volatility_percentile": float(row["volatility_percentile"]),
        "raw_signal_label": raw_signal_label,
        "signal_label": signal_label,
        "buy_blocked": buy_blocked,
        "action_if_held": action_if_held,
        "action_if_not_held": action_if_not_held,
        "reasons": " | ".join(
            [
                trend_reason,
                momentum_reason,
                rsi_reason,
                volatility_reason,
                drawdown_reason,
                return_reason,
                guardrail_reason,
            ]
        ),
    }


def build_signals(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Builds signals for every row with complete, valid price and feature data.

    The output is stateless: it describes what to do both when already holding
    the stock and when currently flat. Portfolio simulation can select the
    appropriate action later without recalculating the research signal.
    """
    required_columns = {"ticker", "date", "adj_close", "volume", *FEATURE_COLUMNS}
    _validate_columns(inputs, required_columns)

    ranked_inputs = add_volatility_percentile(inputs)
    eligibility_columns = ["adj_close", *FEATURE_COLUMNS, "volatility_percentile"]
    eligible = ranked_inputs.dropna(subset=eligibility_columns).copy()
    eligible = eligible.loc[(eligible["adj_close"] > 0) & (eligible["volume"] > 0)]

    if eligible.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    signal_rows = [score_signal_row(row) for _, row in eligible.iterrows()]
    return pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def save_signals_to_database(signals: pd.DataFrame) -> Path:
    """Atomically replaces the derived signals table contents."""
    _validate_columns(signals, set(SIGNAL_COLUMNS))
    initialise_signals_table()

    rows = signals.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    rows["buy_blocked"] = rows["buy_blocked"].astype(int)
    rows = rows[SIGNAL_COLUMNS]

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("DELETE FROM signals")
        conn.executemany(
            """
            INSERT INTO signals (
                ticker, date, trend_score, momentum_score, rsi_score,
                volatility_score, drawdown_score, signal_score,
                volatility_percentile, raw_signal_label, signal_label,
                buy_blocked, action_if_held, action_if_not_held, reasons
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows.itertuples(index=False, name=None),
        )

    return DATABASE_PATH


def calculate_and_save_signals() -> pd.DataFrame:
    """Runs the complete repeatable feature-to-signal database pipeline."""
    inputs = load_signal_inputs()
    signals = build_signals(inputs)
    save_signals_to_database(signals)
    return signals


if __name__ == "__main__":
    generated_signals = calculate_and_save_signals()
    print(f"Saved {len(generated_signals)} signals to {DATABASE_PATH}")
