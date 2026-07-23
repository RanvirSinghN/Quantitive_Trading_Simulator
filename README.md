# Quantitive_Trading_Simulator

I am building a quant-style trading assistant/simulator that uses market data, APIs, machine learning, financial theory, and a Streamlit dashboard to analyse and simulate trades. The goal is to build a strong portfolio project for while properly understanding the backend, modelling logic, and risk management.

I want to learn how quantitative trading systems work from the ground up. The aim is not to create a magic stock picker or an automated trading bot. It is a research project where I can test ideas on historical data, understand why a signal was produced, and gradually build better portfolio and risk-management tools.

## Current Build

The project currently supports historical backtesting across multiple stocks using market data from Yahoo Finance with a basic streamlit dashboard.

### What it does so far

The simulator can:

- Download historical daily stock data from Yahoo Finance
- Clean and validate the market data before using it
- Store prices, features, and signals in a temporary SQLite database
- Calculate common technical indicators for each stock
- Turn those indicators into an explainable bullish, bearish, or neutral signal
- Apply basic risk checks before allowing a new buy
- Simulate trades using the next trading day’s opening price
- Split starting cash using basic risk management across multiple stocks
- Report results for each stock and for the combined portfolio
- Keep a trade log showing what happened and why
- Display charts of its trading decisions

### How the simulation works

UPDATE: Now the simulation logic simply happens through the streamlit dashboard when running app.py.

I choose the starting cash, ticker symbols, and simulation start date in `main.py`. The program then runs the following pipeline:

```text
Yahoo Finance data
        ↓
Data cleaning and validation
        ↓
Feature calculation
        ↓
Signal generation
        ↓
Risk guardrails
        ↓
Next-day trade simulation
        ↓
Portfolio results and trade history
```

The starting cash is divided based on the inverse of the 20 day votality for each stock the day simulation starts. Each stock is simulated separately as either fully invested or held in cash, and the results are then combined into one portfolio summary.

To reduce look-ahead bias, a signal created on one trading day is not executed using that same day’s price. Any trade is carried out at the opening price of the next available trading day.

### Signals and features

The strategy currently uses a simple scoring system based on:

- 20-day and 50-day moving averages
- 10-day and 20-day momentum
- 14-day RSI
- 20-day volatility
- Current drawdown from the stock’s previous peak
- One-day returns

These factors are combined into a score and classified as strong bullish, bullish, neutral, bearish, or strong bearish. The simulator also records a plain-English explanation of the factors behind each signal so I can understand the decision instead of treating it like a black box.

### Basic risk controls

The project is long-only at the moment. It does not use leverage, margin, options, or short selling.

A new buy can also be blocked when:

- Volatility is above the 90th percentile of its recent historical range
- The stock is more than 30% below its previous peak
- The one-day loss is unusually large compared with recent volatility

These checks are still fairly basic, but they give me a foundation for building a more complete risk-management layer later.

### Running the simulator

Download the repo however you like and then either:

1) Dashboard:

Run the interactive trading dashboard locally with `streamlit run dashboard.py` to configure input easily and explore historical portfolio simulations. 

Also option to display charts marking when trades occured.

2) Terminal ouput for quick testing: 

 open `main.py` and edit these values:

```python
TOTAL_STARTING_CASH = 20_000.00
TICKERS = ["AAPL", "MSFT", "NVDA"]
SIMULATION_START_DATE = "2020-03-16"
SELL_REMAINING_AT_END = True
```

Then run:

```bash
python main.py
```

The output includes the amount of data created for each ticker, a per-stock performance summary, the overall portfolio result, and the number of simulated trades.

### Project structure

```text
├── main.py                    # Runs the multi-stock simulation on terminal
|── dashboard.py               # Sets up interactive streamlit dashboard to run simulation
├── src/
│   ├── data_loader.py         # Downloads, cleans, and stores price data
│   ├── feature_engineering.py # Calculates technical features
│   ├── strategy.py            # Scores features and creates signals
│   └── backtester.py          # Simulates trades and calculates results
└── data/                      # Local SQLite data and processed files
```

### Testing/Results

Basic test of the simulation by giving the model 20,000 dollars to trade on APPLE, NVIDIA and MICROSOFT stocks from the 16th of March 2020 up to 21st July 2026. 

The model turned the $20,000 into $83,581 giving an overall return of 317.9%. 

### Current limitations

This is an early-stage research simulator, so the results should not be treated as evidence that the strategy would perform the same way in live markets.

The current version does not yet include:

- Bid/ask spreads or liquidity modelling
- Benchmark comparisons
- Detailed performance statistics such as Sharpe ratio and maximum portfolio drawdown
- Paper trading or broker integration
- AI research layer

### What I plan to add next

My next steps are to:

1) Improve the portfolio and risk logic.
2) After the historical simulator is more reliable, I plan to move towards paper trading with a manual approval workflow.
3) Set up live paper trading portfolio with more complex risk management and freedom
4) xplore how I can add in monte carlo simulations to predict future outcomes.

## Disclaimer

This project is for learning and research only. It is not financial advice, and it does not place real trades.
