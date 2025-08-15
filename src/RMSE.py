import numpy as np
from black_scholes import Black_Scholes
from merton import Merton


def rmse_merton(params, df, S0, r, q, T):
    #Parameters to calibrate
    sigma, lam, muJ, sigmaJ = params

    #Create Black–Scholes instance
    bs = Black_Scholes(r=r, q=q, sigma=sigma)

    #Create Merton model instance
    m = Merton(bs_model=bs, lam=lam, muJ=muJ, sigmaJ=sigmaJ, N_max=30)

    #Compute Merton call prices for all strikes in the DataFrame
    prix_model = np.array([m.price_call_Merton(S0, K, T) for K in df["strike"].values])

    #Return RMSE between model prices and observed market mid prices
    return np.sqrt(np.mean((prix_model-df["mid"].values)**2))


def rmse_bs(params, df, S0, r, q, T):
    #Parameter to calibrate
    sigma = params[0]

    #Create Black–Scholes instance
    bs = Black_Scholes(r=r, q=q, sigma=sigma)

    #Compute BS call prices for all strikes in the DataFrame
    prix_model = np.array([bs.price_call_BS(S0, K, T) for K in df["strike"].values])

    #Return RMSE between model prices and observed market mid prices
    return np.sqrt(np.mean((prix_model-df["mid"].values)**2))