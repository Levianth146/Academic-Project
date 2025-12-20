# ==============================================================
# Momentum Signal Backtest vs VN30 Index – Vietnamese Market
# ==============================================================

# Standard imports
import yfinance as yf # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")

ANNUALIZATION_FACTOR = 252  # Trading days per year


def fetch_stock_data(tickers, start_date, end_date):
    """Fetch adjusted close prices for a list of tickers."""
    print("Fetching stock data...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data.dropna(axis=1, how="all", inplace=True)
    print(f"✅ Loaded data for {data.shape[1]} stocks.")
    return data


def fetch_benchmark_data(benchmark_ticker, start_date, end_date):
    """Fetch benchmark index data; fallback to ETF if needed."""
    print(f"Fetching benchmark data for {benchmark_ticker}...")
    data = yf.download(benchmark_ticker, start=start_date, end=end_date)["Close"].squeeze()
    if data.empty:
        print("⚠️ Primary benchmark unavailable. Trying ETF proxy: FUEVFVND.VN")
        data = yf.download("FUEVFVND.VN", start=start_date, end=end_date)["Close"].squeeze()
    print(f"✅ Loaded benchmark data ({len(data)} days).")
    return data


def compute_momentum_signal(stock_prices, lookback_window=10):
    """Compute momentum signal as cumulative return over lookback window."""
    return stock_prices.pct_change(periods=lookback_window)


def compute_forward_returns(stock_prices, forward_window=5):
    """Compute forward returns shifted by forward_window."""
    return stock_prices.pct_change(periods=forward_window).shift(-forward_window)


def run_momentum_backtest(
    momentum_signal,
    forward_returns,
    top_quantile=0.7,
    bottom_quantile=0.3,
    min_stocks=5,
):
    """
    Run long/short momentum backtest.

    Strategy: Long top (1 - top_quantile)%, Short bottom (bottom_quantile)%
    """
    print("Running momentum backtest...")
    strategy_returns = []
    strategy_dates = []

    for date in momentum_signal.index[10:-5]:
        mom_today = momentum_signal.loc[date].dropna()
        if len(mom_today) < min_stocks:
            continue

        high_thresh = mom_today.quantile(top_quantile)
        low_thresh = mom_today.quantile(bottom_quantile)

        high_mom = mom_today[mom_today >= high_thresh].index
        low_mom = mom_today[mom_today <= low_thresh].index

        if len(high_mom) == 0 or len(low_mom) == 0:
            strategy_returns.append(0.0)
            strategy_dates.append(date)
            continue

        ret_high = forward_returns.loc[date, high_mom].mean()
        ret_low = forward_returns.loc[date, low_mom].mean()
        strategy_returns.append(ret_high - ret_low)
        strategy_dates.append(date)

    strategy_series = pd.Series(strategy_returns, index=strategy_dates).dropna()
    print(f"✅ Backtest completed ({len(strategy_series)} trading days).")
    return strategy_series


def calculate_performance_metrics(returns, annualization_factor=ANNUALIZATION_FACTOR):
    """Calculate cumulative return and Sharpe ratio."""
    if returns.empty:
        return {"cum_return": 0.0, "sharpe": 0.0}

    cum_return = (1 + returns).cumprod().iloc[-1] - 1
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(annualization_factor)

    return {"cum_return": cum_return, "sharpe": sharpe}


def plot_performance_comparison(
    strategy_cum_return,
    benchmark_cum_return,
    title="Momentum Strategy vs VN30 Index (2020-2025)",
):
    """Plot cumulative returns of strategy vs benchmark."""
    plt.figure(figsize=(14, 7))
    plt.plot(
        strategy_cum_return.index,
        strategy_cum_return.values,
        label="Momentum Strategy (Long/Short)",
        color="darkgreen",
        linewidth=2.5,
    )
    plt.plot(
        benchmark_cum_return.index,
        benchmark_cum_return.values,
        label="VN30 Index (VN30)",
        color="steelblue",
        linewidth=2.5,
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("Cumulative Return", fontsize=13)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    print("🚀 Starting Momentum Signal Research Project...\n")

    # Parameters
    tickers = [
        "VCB.VN", "VIC.VN", "VHM.VN", "HPG.VN", "MSN.VN",
        "FPT.VN", "VNM.VN", "TCB.VN", "BID.VN", "CTG.VN",
        "MWG.VN", "VRE.VN", "SSI.VN", "NVL.VN", "POW.VN",
        "TPB.VN", "MBB.VN", "GAS.VN", "VIB.VN", "KDH.VN",
    ]
    start_date = "2020-01-01"
    end_date = "2025-12-15"
    benchmark_ticker = "^VN30"

    # Fetch data
    stock_prices = fetch_stock_data(tickers, start_date, end_date)
    benchmark_prices = fetch_benchmark_data(benchmark_ticker, start_date, end_date)

    # Compute signals
    momentum = compute_momentum_signal(stock_prices, lookback_window=10)
    forward_ret = compute_forward_returns(stock_prices, forward_window=5)
    benchmark_daily_ret = benchmark_prices.pct_change()

    # Run backtest
    strategy_returns = run_momentum_backtest(momentum, forward_ret)

    if strategy_returns.empty:
        print("❌ No valid strategy returns to evaluate.")
        return

    # Align dates
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum_full = (1 + benchmark_daily_ret).cumprod()
    common_dates = strategy_cum.index.intersection(benchmark_cum_full.index)
    strategy_cum = strategy_cum.loc[common_dates]
    benchmark_cum = benchmark_cum_full.loc[common_dates]

    # Performance metrics
    strat_metrics = calculate_performance_metrics(strategy_returns)
    benchmark_metrics = calculate_performance_metrics(
        benchmark_daily_ret.loc[common_dates]
    )

    outperformance = strat_metrics["cum_return"] - benchmark_metrics["cum_return"]

    # Print results
    print("\n📊 PERFORMANCE SUMMARY (2020–2025):")
    print(f"- Momentum Strategy: {strat_metrics['cum_return']:.2%}")
    print(f"- VN30 Index:        {benchmark_metrics['cum_return']:.2%}")
    print(f"- Outperformance:    {outperformance:+.2%}")
    print(f"- Strategy Sharpe:   {strat_metrics['sharpe']:.2f}\n")

    # Plot
    plot_performance_comparison(strategy_cum, benchmark_cum)

    # Final insight
    if outperformance > 0:
        print("✅ Strategy outperformed the benchmark.")
    else:
        print(
            "⚠️ Strategy underperformed – possible causes: "
            "retail-dominated market, mean-reversion behavior, or signal decay."
        )

# Run the project
if __name__ == "__main__":
    main()