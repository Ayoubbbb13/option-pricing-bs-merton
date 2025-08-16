import numpy as np
import math

class Merton:
    def __init__(self,bs_model,lam=0.3,muJ=-0.02,sigmaJ=0.10,N_max=30):
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
            sigma_n = np.sqrt(self.bs.sigma**2+(n*self.sigmaJ**2)/T)
            S_n = S0*np.exp(n*self.muJ+0.5*n*self.sigmaJ**2-self.lam*(np.exp(self.muJ+0.5*self.sigmaJ**2)-1)*T)
            w_n = np.exp(-self.lam*T)*(self.lam*T)**n/math.factorial(n)
            Cn += w_n*self.bs.price_call_BS(S_n,K,T,sigma_n)
        return Cn

    #Price a European put option using Merton's jump-diffusion model via put–call parity.
    def price_put_Merton(self,S0,K,T):
        C = self.price_call_Merton(S0,K,T)
        Pn = C-S0*np.exp(-self.bs.q*T)+K*np.exp(-self.bs.r*T)
        return  Pn

    def delta_call_Merton(self, S0, K, T):
        if T <= 0:
            #At maturity, delta is equal to 1, 0 or 0.5
            if S0 > K:
                return 1.0
            if S0 < K:
                return 0.0
            else:
                return 0.5

        #Compute delta for a European call under Merton Jump-diffusion model
        Delta = 0.0
        for n in range(self.N_max):
            sigma_n = np.sqrt(self.bs.sigma**2+(n*self.sigmaJ**2)/T)
            S_n = S0 * np.exp(n * self.muJ + 0.5 * n * self.sigmaJ ** 2 - self.lam * (np.exp(self.muJ + 0.5 * self.sigmaJ ** 2) - 1) * T)
            w_n = np.exp(-self.lam * T) * (self.lam * T) ** n / math.factorial(n)
            delta_bs_n = self.bs.delta_call_BS(S_n, K, T, sigma_override=sigma_n)
            Delta += w_n * delta_bs_n * (S_n / S0)
        return Delta

    def vega_call_Merton(self, S0, K, T):
        if T <= 0:
            #Vega is 0 at maturity
            return 0
        # Compute vega for a European call under Merton Jump-diffusion model
        Vega = 0.0
        for n in range(self.N_max):
            sigma_n = np.sqrt(self.bs.sigma**2+(n*self.sigmaJ**2)/T)
            S_n = S0 * np.exp(n * self.muJ + 0.5 * n * self.sigmaJ ** 2 - self.lam * (np.exp(self.muJ + 0.5 * self.sigmaJ ** 2) - 1) * T)
            w_n = np.exp(-self.lam * T) * (self.lam * T) ** n / math.factorial(n)
            vega_bs_n = self.bs.vega_call_BS(S_n, K, T, sigma_override=sigma_n)
            Vega += w_n * vega_bs_n * (self.bs.sigma / sigma_n)
        return Vega