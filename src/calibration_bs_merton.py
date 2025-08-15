from scipy.optimize import minimize
from market_data import load_clean_options
from RMSE import rmse_bs
from RMSE import rmse_merton
from black_scholes import Black_Scholes
from merton import Merton
from matplotlib import pyplot as plt
import numpy as np

#Load cleaned SPY call options (full set, ITM, OTM), time to maturity (T), and spot price (S0)
calls, calls_itm, calls_otm, T, S0 = load_clean_options()

# Risk-free rate and continuous dividend yield
r, q =  0.03, 0.0

#Initial value for volatility
x0_bs = [0.2]

#Bounds for volatility during optimization
bounds_bs = [(0.05, 2.0)]

#Calibration for Black-Scholes model for call options (full set, OTM, ITM)
res_bs_otm = minimize(rmse_bs, x0_bs, args=(calls_otm, S0, r, q, T), bounds=bounds_bs)
print("BS sigma* OTM=", res_bs_otm.x[0], " RMSE OTM=", res_bs_otm.fun)
res_bs_itm = minimize(rmse_bs, x0_bs, args=(calls_itm, S0, r, q, T), bounds=bounds_bs)
print("BS sigma* ITM=", res_bs_itm.x[0], " RMSE ITM=", res_bs_itm.fun)
res_bs = minimize(rmse_bs, x0_bs, args=(calls, S0, r, q, T), bounds=bounds_bs)
print("BS sigma* =", res_bs.x[0], " RMSE =", res_bs.fun)

#Initial values for [sigma, lambda, mu_J, sigma_J]
x0 = [0.2, 0.5, -0.05, 0.2]

#Bounds for each parameter
bounds_merton = [(0.05, 2.0), (0.00, 2.0), (-0.5, 0.2), (0.05, 0.8)]

#Calibration for Merton jump-diffusion model for call options (full set, OTM, ITM)
res_merton_otm = minimize(rmse_merton, x0, args=(calls_otm, S0, r, q, T), bounds=bounds_merton)
print("Merton [sigma, lam, muJ, sigmaJ] OTM=", res_merton_otm.x, " RMSE =", res_merton_otm.fun)
res_merton_itm = minimize(rmse_merton, x0, args=(calls_itm, S0, r, q, T), bounds=bounds_merton)
print("Merton [sigma, lam, muJ, sigmaJ] ITM=", res_merton_itm.x, " RMSE =", res_merton_itm.fun)
res_merton = minimize(rmse_merton, x0, args=(calls, S0, r, q, T), bounds=bounds_merton)
print("Merton [sigma, lam, muJ, sigmaJ] ITM=", res_merton.x, " RMSE =", res_merton.fun)

def plot_subset(df, S0, T, r, q, res_bs, res_merton, title, nmax=50):
    #Extract strikes K and market mid prices from the dataset
    xK = df["strike"].values
    yM = df["mid"].values

    #Take the calibrated volatility from the BS optimization results
    sigma_bs = res_bs.x[0]
    #Compute Black-Scholes call price for each K
    bs = Black_Scholes(r=r, q=q, sigma=sigma_bs)
    yBS = np.array([bs.price_call_BS(S0, K, T) for K in xK])

    #Take the calibrated Merton parameters
    sigma_m, lam_m, muJ_m, sigmaJ_m = res_merton.x
    # Compute Merton call prices for each strike
    bs_for_m = Black_Scholes(r=r, q=q, sigma=sigma_m)
    m = Merton(bs_for_m, lam=lam_m, muJ=muJ_m, sigmaJ=sigmaJ_m, N_max=nmax)
    yMT = np.array([m.price_call_Merton(S0, K, T) for K in xK])

    #Plot market mid prices, BS prices, and Merton prices as a function of K
    plt.figure()
    plt.scatter(xK, yM, label="market (mid)")
    plt.plot(xK, yBS, label=f"BS (param={round(sigma_bs, 2)})")
    plt.plot(xK, yMT, label= f"Merton (params={round(sigma_m, 2)},{round(lam_m, 2)},{round(muJ_m, 2)},{round(sigmaJ_m, 2)})")
    plt.xlabel("Strike K")
    plt.ylabel("call price")
    plt.title(title)
    plt.legend()
    plt.show()


plot_subset(calls_otm, S0, T, r, q, res_bs_otm, res_merton_otm, title="OTM", nmax=30)
plot_subset(calls_itm, S0, T, r, q, res_bs_itm, res_merton_itm, title="ITM", nmax=30)
plot_subset(calls,     S0, T, r, q, res_bs, res_merton, title="All calls", nmax=30)