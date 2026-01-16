# 📈 Modern Portfolio Theory (Markowitz Optimization)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![SciPy](https://img.shields.io/badge/SciPy-Optimization-red)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📌 Overview
This project implements **Mean-Variance Optimization** (Modern Portfolio Theory) to construct an optimal investment portfolio. By analyzing historical data of S&P 500 stocks, the algorithm identifies the asset allocation that maximizes the **Sharpe Ratio** (risk-adjusted return).

The program generates an **Efficient Frontier**, visualizing the trade-off between Risk (Volatility) and Return.

## 🧮 The Math
The optimization problem is defined as minimizing the negative Sharpe Ratio:

$$\text{Maximize } S_p = \frac{E(R_p) - R_f}{\sigma_p}$$

Where portfolio risk $\sigma_p$ is calculated using the Covariance Matrix $\Sigma$ and weight vector $w$:

$$\sigma_p = \sqrt{w^T \Sigma w}$$

Subject to constraints:
1. $\sum w_i = 1$ (Fully invested)
2. $0 \leq w_i \leq 1$ (No short selling)

## 🛠️ Tech Stack
- **Python**: Core logic
- **SciPy**: `minimize` function (SLSQP algorithm) for quadratic programming
- **Pandas/NumPy**: Data manipulation and matrix algebra
- **YFinance**: Real-time market data extraction
- **Matplotlib**: Visualization of the Efficient Frontier

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/moonthesir/portfolio-optimization.git](https://github.com/moonthesir/portfolio-optimization.git)
