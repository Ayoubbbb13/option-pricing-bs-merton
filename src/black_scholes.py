import numpy as np
from scipy.stats import norm

class Black_Scholes:
    def __init__(self,r=0.03,q=0.0,sigma=0.05):
        #Risk-free rate
        self.r = r
        #Continuous dividend yield
        self.q = q
        #Private attribute for volatility
        self._sigma = sigma

    @property
    def sigma(self):
        #getter for volatility
        return self._sigma

    @sigma.setter
    def sigma(self,value):
        #Setter with validation: sigma must be positive
        if value < 0:
            raise ValueError("sigma must be positive")
        self._sigma = value

    def price_call_BS(self,S0,K,T,sigma_override=None):
        #Use the object's sigma unless an override is provided
        if sigma_override is None:
            sigma = self.sigma
        else:
            sigma = sigma_override
        sigma = max(sigma, 10**-6)

        #Price a european call
        d1 = (np.log(S0/K)+(self.r-self.q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)
        C = S0*np.exp(-self.q*T)*norm.cdf(d1)-K*np.exp(-self.r*T)*norm.cdf(d2)
        return C

    def price_put_BS(self,S0,K,T,sigma_override=None):
        #Price a european put via Call-Put parity
        C = self.price_call_BS(S0,K,T,sigma_override)
        P = C-S0*np.exp(-self.q*T)+K*np.exp(-self.r*T)
        return P

    def delta_call_BS(self, S0,K,T,sigma_override=None):
        #Use the object's sigma unless an override is provided
        if sigma_override is None:
            sigma = self.sigma
        else:
            sigma = sigma_override
        if T <= 0:
            #At maturity delta is equal to 1, 0 or 0.5
            if S > K:
                return 1
            if S < K:
                return 0
            else:
                return 0.5
        sigma = max(sigma, 10**-6)

        #Compute delta for a European call under Black-Scholes model
        d1 = (np.log(S0/K)+(self.r-self.q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return np.exp(-self.q*T)*norm.cdf(d1)

    def vega_call_BS(self, S0, K, T, sigma_override=None):
        if sigma_override is None:
            sigma = self.sigma
        else:
            sigma = sigma_override
        sigma = max(sigma, 10**-6)
        if T <= 0:
            #At maturity delta is equal to 1, 0 or 0.5
            return 0
        #Compute vega for a European call under Black-Scholes model
        d1 = (np.log(S0 / K) + (self.r - self.q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S0 * np.exp(-self.q*T)*np.sqrt(T)*(1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*d1**2)
