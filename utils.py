import pandas as pd
import numpy as np
from pykrx import stock
import streamlit as st
from arch import arch_model
import pyupbit
import datetime

# 1. 수치 해석 모델 (Numerical Analysis)
def get_numerical_analysis(prices, h=1):
    if len(prices) < 10:
        return {"euler": prices[-1], "rk4": prices[-1], "newton": prices[-1], "simpson": 0}
    
    y = prices
    v = (y[-1] - y[-2]) 
    a = (y[-1] - y[-2]) - (y[-2] - y[-3])
    taylor_2nd = y[-1] + (h * v) + (0.5 * (h**2) * a)
    
    k1 = y[-1] - y[-2]          
    k2 = (y[-1] - y[-3]) / 2    
    k3 = (y[-2] - y[-4]) / 2    
    k4 = (y[-1] - y[-5]) / 4 if len(y) > 5 else k1
    rk4_val = y[-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    f_x = y[-1] - y[-2]
    f_prime_x = (y[-1] - 2*y[-2] + y[-3])
    if f_prime_x == 0: f_prime_x = 1
    newton_val = y[-1] - (f_x / f_prime_x) * h

    simpson_energy = (h/3) * (y[-3] + 4*y[-2] + y[-1])
    simpson_adj = (simpson_energy / (y[-1] * 3))
    
    return {"euler": taylor_2nd, "rk4": rk4_val, "newton": newton_val, "simpson": simpson_adj}

# 2. 기술적 지표 계산 (Technical Indicators)
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
    df['RSI'] = df['RSI'].fillna(50) 

    df['거래량_변동률'] = df['거래량'].pct_change() * 100
    df['거래량_변동률'] = df['거래량_변동률'].replace([float('inf'), float('-inf')], 0).fillna(0)

    df['등락률'] = df['종가'].pct_change() * 100
    df['등락률'] = df['등락률'].fillna(0)
    
    return df

# 3. 가상화폐 티커 확보
@st.cache_data
def get_coin_tickers():
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        return {t.split('-')[1]: t for t in tickers}
    except Exception:
        return {"BTC": "KRW-BTC"}

# 4. [완벽 해결] 주식 종목 리스트 확보
@st.cache_data(ttl=3600) 
def get_tickers():
    search_dt = datetime.datetime.now()
    
    for _ in range(15):
        target_date = search_dt.strftime("%Y%m%d")
        try:
            # 타임아웃을 막기 위해 코스피, 코스닥만 호출 (코넥스 제외)
            kse = stock.get_market_ticker_list(target_date, market="KOSPI")
            kdq = stock.get_market_ticker_list(target_date, market="KOSDAQ")
            
            all_tickers = list(set(kse + kdq))
            
            # 500개 이상만 잡혀도 정상 데이터로 인정 (기준 완화)
            if all_tickers and len(all_tickers) > 500:
                ticker_dict = {}
                for t in all_tickers:
                    try:
                        name = stock.get_market_ticker_name(t)
                        if name:
                            ticker_dict[name] = t
                    except:
                        continue # 에러 나는 불량 종목은 패스
                
                if len(ticker_dict) > 500:
                    return ticker_dict
        except Exception:
            pass
        
        search_dt -= datetime.timedelta(days=1)
    
    # 🚨 여기에 걸렸다면 KRX 서버 접근 차단 또는 인터넷 문제입니다.
    return {
        "서버접근_실패_상태": "000000", "삼성전자": "005930", "SK하이닉스": "000660",
        "에코프로": "086520", "HLB": "028300", "한일시멘트": "300720", "테크윙": "089030"
    }

# 5. 변동성 모델 (GARCH Family)
def get_volatility_models(prices):
    prices = np.where(prices <= 0, 1e-9, prices)
    returns = np.diff(np.log(prices)) * 100
    returns = np.nan_to_num(returns, nan=0.0, posinf=30.0, neginf=-30.0)
    returns = np.clip(returns, -35.0, 35.0) 
    
    train_data = returns[-500:] if len(returns) > 500 else returns 
    baseline_vol = np.std(train_data)
    if baseline_vol == 0: baseline_vol = 1e-9
    max_vol_limit = baseline_vol * 10.0 
    
    try:
        egarch_m = arch_model(train_data, vol='EGARCH', p=1, o=1, q=1, dist='t', rescale=True)
        egarch_res = egarch_m.fit(disp='off', show_warning=False)
        egarch_var = egarch_res.forecast(horizon=1).variance.iloc[-1].values[0]
        egarch_vol = np.sqrt(egarch_var) if egarch_var > 0 else baseline_vol
        
        gjr_m = arch_model(train_data, p=1, o=1, q=1, dist='t', rescale=True)
        gjr_res = gjr_m.fit(disp='off', show_warning=False)
        gjr_var = gjr_res.forecast(horizon=1).variance.iloc[-1].values[0]
        gjr_vol = np.sqrt(gjr_var) if gjr_var > 0 else baseline_vol
        
    except Exception:
        egarch_vol = baseline_vol
        gjr_vol = baseline_vol
        
    return {
        "egarch": float(np.clip(egarch_vol, 0, max_vol_limit)),
        "gjr_garch": float(np.clip(gjr_vol, 0, max_vol_limit))
    }
