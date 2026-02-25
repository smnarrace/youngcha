import pandas as pd
import numpy as np
from pykrx import stock
import streamlit as st
from arch import arch_model

def get_numerical_analysis(prices, h=1):
    if len(prices) < 10: return {"euler": prices[-1], "rk4": prices[-1], "newton": prices[-1], "simpson": 0}
    y = prices
    slope_simple = (y[-1] - y[-h-1]) / h if len(y) > h else (y[-1] - y[-2])
    euler = y[-1] + h * slope_simple
    
    k1 = y[-1] - y[-2]          # 최근 1일 변화량
    k2 = (y[-1] - y[-3]) / 2    # 최근 2일 평균 변화량
    k3 = (y[-2] - y[-4]) / 2    # 전일 기준 2일 평균 변화량
    k4 = (y[-1] - y[-5]) / 4 if len(y) > 5 else k1
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


# utils.py 의 변동성 함수 교체
def get_volatility_models(prices):
    # 1. 0이나 음수 가격 방지 (오류 원천 차단)
    prices = np.where(prices <= 0, 1e-9, prices)
    
    # 2. 로그 수익률 계산 (안전한 방식)
    returns = np.diff(np.log(prices)) * 100
    
    # 3. 무한대(inf)나 결측치(NaN)를 0으로 강제 치환 [★핵심]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    train_data = returns[-500:] if len(returns) > 500 else returns 
    
    try:
        # EGARCH 모델
        egarch_m = arch_model(train_data, vol='EGARCH', p=1, o=1, q=1, dist='t', rescale=False)
        egarch_res = egarch_m.fit(disp='off', show_warning=False)
        egarch_var = egarch_res.forecast(horizon=1).variance.iloc[-1].values[0]
        egarch_vol = np.sqrt(egarch_var) if egarch_var > 0 else 0
        
        # GJR-GARCH 모델
        gjr_m = arch_model(train_data, p=1, o=1, q=1, dist='t', rescale=False)
        gjr_res = gjr_m.fit(disp='off', show_warning=False)
        gjr_var = gjr_res.forecast(horizon=1).variance.iloc[-1].values[0]
        gjr_vol = np.sqrt(gjr_var) if gjr_var > 0 else 0
        
    except Exception as e:
        # 모델 적합에 실패하여 에러가 날 경우, 단순 표준편차로 대책 마련
        egarch_vol = np.std(train_data)
        gjr_vol = np.std(train_data)
        
    # 4. 변동성이 비정상적으로 폭발(예: 300%)하는 것을 막기 위해 최대치(예: 30%)로 제한
    return {
        "egarch": float(np.clip(egarch_vol, 0, 30.0)),
        "gjr_garch": float(np.clip(gjr_vol, 0, 30.0))
    }
