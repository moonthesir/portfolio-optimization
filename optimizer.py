import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# --- 1. CONFIGURATION ---
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'SPY']
start_date = '2020-01-01'
end_date = '2024-01-01'
risk_free_rate = 0.04  # 4%

# --- 2. GET DATA (FIXED) ---
print(f"Downloading data for: {tickers}...")
# We use auto_adjust=True to handle splits/dividends automatically
# This prevents the 'Adj Close' key error
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
data = raw_data['Close']

# Calculate Daily Returns
returns = data.pct_change().dropna()

# --- 3. OPTIMIZATION FUNCTIONS ---
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_var = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

# --- 4. RUN OPTIMIZATION ---
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(tickers)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets,]

print("Optimizing portfolio weights...")
opt_results = optimize.minimize(neg_sharpe_ratio, init_guess,
                                args=(mean_returns, cov_matrix, risk_free_rate),
                                method='SLSQP', bounds=bounds, constraints=constraints)

# --- 5. RESULTS ---
best_weights = opt_results.x
best_ret, best_vol = portfolio_performance(best_weights, mean_returns, cov_matrix)
best_sharpe = (best_ret - risk_free_rate) / best_vol

print("\n--- OPTIMAL PORTFOLIO ---")
print(f"Expected Annual Return: {best_ret:.2%}")
print(f"Annual Volatility:      {best_vol:.2%}")
print(f"Sharpe Ratio:           {best_sharpe:.2f}")
print("\nAllocation:")
for ticker, weight in zip(tickers, best_weights):
    print(f"  {ticker}: {weight:.2%}")

# --- 6. PLOT EFFICIENT FRONTIER ---
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    results[0,i] = p_std
    results[1,i] = p_ret
    results[2,i] = (p_ret - risk_free_rate) / p_std

plt.figure(figsize=(10, 6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(best_vol, best_ret, marker='*', color='red', s=300, label='Maximum Sharpe')
plt.title('Efficient Frontier (Mean-Variance Optimization)')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend()
plt.show()
