import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pandas as pd

def get_daily_prices(symbol, start_date, end_date):
    """
    Retrieve the daily adjusted closing prices for a stock symbol using yfinance.
    If "Adj Close" is not available, it falls back to "Close".
    Returns a pandas Series indexed by date.
    """
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        print(f"No data found for {symbol} between {start_date} and {end_date}.")
        return None
    # Use "Adj Close" if available, otherwise "Close"
    if "Adj Close" in data.columns:
        series = data["Adj Close"]
    else:
        series = data["Close"]
    # Squeeze to ensure it's a Series
    series = series.squeeze()
    return series

def align_series(series1, series2):
    """
    Align two pandas Series on their common dates and return their values as numpy arrays.
    """
    df = series1.to_frame("s1").join(series2.to_frame("s2"), how="inner")
    return df["s1"].values, df["s2"].values

def main():
    # Define the stock symbols and the date range.
    symbols = ["AAPL", "cost", "f", "lazr", "IIPR", "NFLX", "GM", "tgt", "WMT"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    # Retrieve prices for each stock.
    stock_prices = {}
    for symbol in symbols:
        print("Fetching data for", symbol)
        series = get_daily_prices(symbol, start_date, end_date)
        if series is None:
            print("No data for", symbol)
        stock_prices[symbol] = series
    
    # Compute Pearson correlations for each unique pair.
    correlations = {}
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            s1 = stock_prices[symbols[i]]
            s2 = stock_prices[symbols[j]]
            if s1 is None or s2 is None:
                corr = None
            else:
                aligned1, aligned2 = align_series(s1, s2)
                if len(aligned1) == 0 or len(aligned2) == 0:
                    corr = None
                else:
                    corr = np.corrcoef(aligned1, aligned2)[0, 1]
            correlations[(symbols[i], symbols[j])] = corr
            print(f"Correlation between {symbols[i]} and {symbols[j]}: {corr}")
    
    # Arrange stocks in a circle.
    n = len(symbols)
    radius = 10
    positions = {}
    for i, symbol in enumerate(symbols):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions[symbol] = (x, y)
    
    # Set up the first plot (circular branch plot).
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    
    # Plot each stock as a point with its label.
    for symbol, (x, y) in positions.items():
        ax.scatter(x, y, color='blue', s=100, zorder=5)
        ax.text(x, y, symbol, fontsize=12, ha='center', va='center', color='white', zorder=10)
    
    # Draw white branches between stocks and annotate with correlation values.
    for (sym1, sym2), corr in correlations.items():
        x1, y1 = positions[sym1]
        x2, y2 = positions[sym2]
        ax.plot([x1, x2], [y1, y2], color='white', lw=1, alpha=0.5, zorder=1)
        if corr is not None:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label = "r={:.2f}".format(corr)
            ax.text(mid_x, mid_y, label, fontsize=12, ha='center', va='center', 
                    color='red', zorder=30,
                    bbox=dict(facecolor='black', alpha=1.0, edgecolor='none', pad=2))
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Pairwise Pearson Correlations Among 10 Stocks", fontsize=16, color='white', zorder=40)
    plt.tight_layout()
    plt.show()
    
    # -------------------------------
    # Build and display the correlation matrix
    # -------------------------------
    
    # Create a DataFrame with the price series for all stocks that have data.
    price_df = pd.DataFrame({symbol: series for symbol, series in stock_prices.items() if series is not None})
    
    # Compute the correlation matrix.
    corr_matrix = price_df.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Plot the correlation matrix as a heatmap.
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")
    
    cax = ax2.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(corr_matrix.columns)))
    ax2.set_yticks(range(len(corr_matrix.index)))
    ax2.set_xticklabels(corr_matrix.columns, rotation=90, color="white")
    ax2.set_yticklabels(corr_matrix.index, color="white")
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    plt.title("Correlation Matrix", color="white")
    fig2.colorbar(cax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
