import numpy as np
import math

class Merton:
    def __init__(self, bs_model, lam=0.3, muJ=-0.02, sigmaJ=0.10, N_max=30):
        #Instance of Black-Scholes
        self.bs = bs_model
        #Jump intensity lambda (average number of jumps per year)
        self.lam = lam
        #Mean log jump size
        self.muJ = float(muJ)
        #Standard deviation of log jump size
        self.sigmaJ = float(sigmaJ)
        #Limit for the Poisson summation
        self.N_max = int(N_max)

    #Price a European call option using Merton's jump-diffusion formula
    def price_call_Merton(self, S0, K, T):
        Cn = 0.0
        for n in range(self.N_max):
            sigma_n = np.sqrt(self.bs.sigma ** 2 + (n * self.sigmaJ ** 2) / T)
            S_n = S0 * np.exp(
                n * self.muJ + 0.5*n*self.sigmaJ**2 - self.lam * (np.exp(self.muJ + 0.5 * self.sigmaJ ** 2) - 1) * T)
            w_n = np.exp(-self.lam * T) * (self.lam * T) ** n / math.factorial(n)
            Cn += w_n * self.bs.price_call_BS(S_n, K, T, sigma_n)
        return Cn

    #Price a European put option using Merton's jump-diffusion model via put–call parity.
    def price_put_Merton(self, S0, K, T):
        C = self.price_call_Merton(S0, K, T)
        Pn = C - S0 * np.exp(-self.bs.q * T) + K * np.exp(-self.bs.r * T)
        return  Pn
