# Merton Jump-Diffusion Model – Pricing & Calibration
This repository implements the Black–Scholes and Merton Jump-Diffusion models to compare their option pricing performance using S&P 500 option data.
## Main steps
Data preparation: extraction and cleaning of S&P 500 option data (removal of illiquid quotes, computation of mid-prices, split into ITM/OTM).
Model implementation: object-oriented Python classes for Black–Scholes and Merton models (pricing formulas and Greeks).
Calibration: parameter estimation by minimizing RMSE between market and model prices.
Analysis:
Comparison of call option prices across strikes
Implied volatility smile from market vs Merton prices
Sensitivity analysis (Delta, Vega) under both models
