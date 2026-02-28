import pandas as pd
import numpy as np
from pykrx import stock
import streamlit as st
from arch import arch_model
import pyupbit
import datetime

# 1. 수치 해석 모델 (Numerical Analysis)
def get_numerical_analysis(prices, h=1):
    """
    Euler, RK4, Newton, Simpson 등의 수치해석 기법을 이용한 가격 예측
    """
    if len(prices) < 10:
        return {"euler": prices[-1], "rk4": prices[-1], "newton": prices[-1], "simpson": 0}
    
    y = prices
    
    # 테일러 2차 (Euler 기반 확장)
    v = (y[-1] - y[-2]) 
    a = (y[-1] - y[-2]) - (y[-2] - y[-3])
    taylor_2nd = y[-1] + (h * v) + (0.5 * (h**2) * a)
    
    # Runge-Kutta 4th Order (RK4) 근사
    k1 = y[-1] - y[-2]          
    k2 = (y[-1] - y[-3]) / 2    
    k3 = (y[-2] - y[-4]) / 2    
    k4 = (y[-1] - y[-5]) / 4 if len(y) > 5 else k1
    rk4_val = y[-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Newton-Raphson 기반 예측
    f_x = y[-1] - y[-2]
    f_prime_x = (y[-1] - 2*y[-2] + y[-3])
    if f_prime_x == 0: f_prime_x = 1
    newton_val = y[-1] - (f_x / f_prime_x) * h

    # Simpson's Rule (에너지/면적 보정 지표)
    simpson_energy = (h/3) * (y[-3] + 4*y[-2] + y[-1])
    simpson_adj = (simpson_energy / (y[-1] * 3))
    
    return {"euler": taylor_2nd, "rk4": rk4_val, "newton": newton_val, "simpson": simpson_adj}

# 2. 기술적 지표 계산 (Technical Indicators)
def calculate_indicators(df):
    """
    이동평균선, 볼린저 밴드, RSI, 거래량 변동률 등 계산
    """
    # 이동평균선 및 볼린저 밴드
    df['MA20'] = df['종가'].rolling(window=20).mean()
    std = df['종가'].rolling(window=20).std()
    df['BB_U'] = df['MA20'] + (std * 2)
    df['BB_L'] = df['MA20'] - (std * 2)
    
    # RSI (Relative Strength Index)
    delta = df['종가'].diff()
    up = delta.clip(lower=0).rolling(window=14).mean()
    down = delta.clip(upper=0).abs().rolling(window=14).mean()
    rs = up / (down + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) 

    # 거래량 변동률 (%)
    df['거래량_변동률'] = df['거래량'].pct_change() * 100
    df['거래량_변동률'] = df['거래량_변동률'].replace([float('inf'), float('-inf')], 0).fillna(0)

    # 등락률 (%) - 모델의 학습 타겟
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

# 4. [핵심] 주식 종목 리스트 확보 (에코프로, HLB 등 전 종목 지원)
@st.cache_data(ttl=3600) 
def get_tickers():
    """
    오늘부터 최대 15일 전까지 뒤져서 실제 데이터가 있는 날의 전체 종목 리스트를 가져옴
    """
    search_dt = datetime.datetime.now()
    
    for _ in range(15):
        target_date = search_dt.strftime("%Y%m%d")
        try:
            # 코스피/코스닥 전체 종목 코드를 가져옵니다.
            tickers = stock.get_market_ticker_list(target_date, market="ALL")
            
            # 종목이 1000개 이상은 되어야 정상적인 시장 리스트로 간주합니다.
            if tickers and len(tickers) > 1000:
                return {stock.get_market_ticker_name(t): t for t in tickers}
        except:
            pass
        # 데이터가 없으면(토, 일, 공휴일 등) 하루 전으로 이동해서 다시 시도
        search_dt -= datetime.timedelta(days=1)
    
    # 모든 탐색이 실패했을 때의 최후 보루 (백업 리스트)
    return {
        "삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380", 
        "NAVER": "035420", "카카오": "035720", "에코프로": "086520", "HLB": "028300"
    }

# 5. 변동성 모델 (GARCH Family)
def get_volatility_models(prices):
    """
    EGARCH 및 GJR-GARCH 모델을 이용한 변동성 예측
    """
    # 가격 예외 처리
    prices = np.where(prices <= 0, 1e-9, prices)
    
    # 로그 수익률 계산 (%)
    returns = np.diff(np.log(prices)) * 100
    
    # 윈저라이징 (Winsorizing): 극단치 보정
    returns = np.nan_to_num(returns, nan=0.0, posinf=30.0, neginf=-30.0)
    returns = np.clip(returns, -35.0, 35.0) 
    
    # 학습용 데이터 (최근 500일)
    train_data = returns[-500:] if len(returns) > 500 else returns 
    
    # 기본 변동성 (Standard Deviation)
    baseline_vol = np.std(train_data)
    if baseline_vol == 0: baseline_vol = 1e-9
    max_vol_limit = baseline_vol * 10.0 
    
    try:
        # EGARCH 모델 (비대칭 변동성)
        egarch_m = arch_model(train_data, vol='EGARCH', p=1, o=1, q=1, dist='t', rescale=True)
        egarch_res = egarch_m.fit(disp='off', show_warning=False)
        egarch_var = egarch_res.forecast(horizon=1).variance.iloc[-1].values[0]
        egarch_vol = np.sqrt(egarch_var) if egarch_var > 0 else baseline_vol
        
        # GJR-GARCH 모델
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
