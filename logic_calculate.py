import pandas as pd
import numpy as np
from pykrx import stock
import streamlit as st

def get_numerical_analysis(prices, h=1):
    if len(prices) < 10: return {"euler": prices[-1], "rk4": prices[-1], "newton": prices[-1], "simpson": 0}
    y = prices
    slope = (y[-1] - y[-h-1]) / h if len(y) > h else (y[-1] - y[-2])
    
    euler = y[-1] + h * slope
    k1, k2, k3, k4 = slope, slope * 1.01, slope * 1.01, slope * 1.02
    rk4 = y[-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    f_x = y[-1] - y[-2]
    f_prime_x = (y[-1] - 2*y[-2] + y[-3])
    if f_prime_x == 0: f_prime_x = 1
    newton = y[-1] - (f_x / f_prime_x) * h

    simpson_energy = (h/3) * (y[-3] + 4*y[-2] + y[-1])
    simpson_adj = (simpson_energy / (y[-1] * 3))
    
    return {"euler": euler, "rk4": rk4, "newton": newton, "simpson": simpson_adj}

def calculate_indicators(df):
    df['MA20'] = df['종가'].rolling(window=20).mean()
    std = df['종가'].rolling(window=20).std()
    df['BB_U'] = df['MA20'] + (std * 2)
    df['BB_L'] = df['MA20'] - (std * 2)
    
    delta = df['종가'].diff()
    up = delta.clip(lower=0).rolling(window=14).mean()
    down = delta.clip(upper=0).abs().rolling(window=14).mean()
    rs = up / (down + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_data
def get_tickers():
    tickers = stock.get_market_ticker_list()
    return {stock.get_market_ticker_name(t): t for t in tickers}