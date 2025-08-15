from market_data import load_clean_options
from matplotlib import pyplot as plt
from merton import Merton
from black_scholes import Black_Scholes
from scipy.optimize import brentq


calls, calls_itm, calls_otm, T, S0 = load_clean_options()
r, q =  0.03, 0.0

#Function to compute implied volatility for a call
def implied_vol_call(price, S0, K, T, r, q, bs_model):
    def f(sigma):
        bs_model.sigma = sigma
        return bs_model.price_call_BS(S0, K, T)-price
    return brentq(f, a=0, b=5.0)

#Sort OTM calls by strike and keep strike and mid price columns
df_iv = calls_otm.sort_values("strike")[["strike","mid"]].copy()

#Black–Scholes model instance
bs = Black_Scholes(r=r, q=q, sigma=0.2)

#Compute market implied volatilities for BS from observed mid prices
df_iv["iv_bs_mkt"] = [implied_vol_call(p, S0, K, T, r, q, bs) for p, K in zip(df_iv["mid"], df_iv["strike"])]

# Calibrated Merton parameters
sigma_m, lam_m, muJ_m, sigmaJ_m = 0.08018708 ,0.54193024, -0.1819641,   0.15708393

#Merton model instance
merton = Merton(Black_Scholes(r=r, q=q, sigma=sigma_m),lam=lam_m, muJ=muJ_m, sigmaJ=sigmaJ_m, N_max=40)

#Compute Merton model prices for each strike
df_iv["price_merton"] = [merton.price_call_Merton(S0, K, T) for K in df_iv["strike"]]

# Compute implied volatilities for Merton model
df_iv["iv_merton"] = [implied_vol_call(p, S0, K, T, r, q, bs) for p, K in zip(df_iv["price_merton"], df_iv["strike"])]

#Plot implied volatility smile
plt.figure()
plt.plot(df_iv["strike"], df_iv["iv_bs_mkt"], label="IV BS", marker="o")
plt.plot(df_iv["strike"], df_iv["iv_merton"], label="IV Merton", marker="s")
plt.xlabel("Strike K")
plt.ylabel("Implied volatility")
plt.legend()
plt.show()