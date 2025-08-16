import pandas as pd
from matplotlib import pyplot as plt
from merton import Merton
from black_scholes import Black_Scholes

#Load option datasets
calls = pd.read_csv("calls_SPY_2025-12-19.csv")
calls_itm = pd.read_csv("calls_ITM_SPY_2025-12-19.csv")
calls_otm = pd.read_csv("calls_OTM_SPY_2025-12-19.csv")
T = calls['T'].values[0]
S0 = calls['S0'].values[0]

#Strike values for plotting Greeks
strike_calls = calls["strike"].values

#Compute Black-Scholes delta for each strike
sigma_bs = 0.14740306344137222
bs = Black_Scholes()
delta_call_bs = bs.delta_call_BS(S0,strike_calls,T,sigma_bs)

#Compute Merton's delta for each strike
bs_merton = Black_Scholes(sigma = 0.08855354)
lam, muJ, sigmaJ,= 0.40475354,-0.20440263,0.16154693
merton = Merton(bs_merton, lam,muJ,sigmaJ)
delta_call_merton = merton.delta_call_Merton(S0,strike_calls,T)

#Compute Black-Schole vega for each strike
vega_call_bs = bs.vega_call_BS(S0,strike_calls,T,sigma_bs)
#Compute Merton's vega for each strike
vega_call_merton = merton.vega_call_Merton(S0,strike_calls,T)

#Plot delta comparison
plt.figure()
plt.plot(strike_calls,delta_call_bs,label="delta BS")
plt.plot(strike_calls,delta_call_merton,label="delta Merton")
plt.xlabel("strike")
plt.ylabel("delta")
plt.legend()
plt.show()

#Plot vega comparison
plt.figure()
plt.plot(strike_calls,vega_call_bs,label="vega BS")
plt.plot(strike_calls,vega_call_merton,label="vega Merton")
plt.xlabel("strike")
plt.ylabel("vega")
plt.legend()
plt.show()


