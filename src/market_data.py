import pandas as pd
import yfinance as yf

def load_clean_options(ticker="SPY", maturity = "2025-12-19"):
    #Download option chain data for the given maturity
    asset = yf.Ticker(ticker)
    chain = asset.option_chain(maturity)

    #Extract call options from the chain
    calls = chain.calls.copy()
    calls["type"] = "call"

    #Keep only relevant columns
    calls = calls[["type", "contractSymbol", "strike", "lastPrice", "bid", "ask","volume", "openInterest", "impliedVolatility", "lastTradeDate", "inTheMoney"]]

    #Filter out options with non-positive bid or ask prices
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()

    #Compute the mid price
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2

    #Keep only options with either volume > 0 or open interest > 0
    calls = calls[((calls["volume"] > 0) | (calls["openInterest"] > 0))]

    #Download the most recent historical data to get the spot price
    hist = asset.history(period="1d")
    S0 = hist["Close"].iloc[-1]

    #Compute time to maturity in years
    snap_time = hist.index[-1].tz_localize(None)
    T_exp = pd.Timestamp(maturity)
    T = (T_exp - snap_time).total_seconds()/(365*24*3600)

    #Separate calls into in-the-money and out-of-the-money
    calls_itm = calls[calls["strike"] < S0].copy()
    calls_otm = calls[calls["strike"] > S0].copy()

    return  calls, calls_itm, calls_otm, T, S0

#Save cleaned call option data for SPY (full set, ITM, and OTM) with time to maturity T and spot price S0
calls, calls_itm, calls_otm,T,S0 = load_clean_options()
for df in (calls, calls_itm, calls_otm):
    df["T"]  = T
    df["S0"] = S0

calls.to_csv("calls_SPY_2025-12-19.csv", index=False)
calls_itm.to_csv("calls_itm_SPY_2025-12-19.csv", index=False)
calls_otm.to_csv("calls_otm_SPY_2025-12-19.csv", index=False)

