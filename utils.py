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
    # 1. 이동평균선 및 볼린저 밴드 (기존 유지)
    df['MA20'] = df['종가'].rolling(window=20).mean()
    std = df['종가'].rolling(window=20).std()
    df['BB_U'] = df['MA20'] + (std * 2)
    df['BB_L'] = df['MA20'] - (std * 2)
    
    # 2. RSI (Relative Strength Index) 계산
    delta = df['종가'].diff()
    up = delta.clip(lower=0).rolling(window=14).mean()
    down = delta.clip(upper=0).abs().rolling(window=14).mean()
    rs = up / (down + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) # 초기값 중립(50) 처리

    # 3. [추가] 거래량 변동률 (%) - AI의 '에너지' 피처
    # 전일 대비 거래량이 얼마나 터졌는지 계산합니다.
    df['거래량_변동률'] = df['거래량'].pct_change() * 100
    df['거래량_변동률'] = df['거래량_변동률'].replace([float('inf'), float('-inf')], 0).fillna(0)

    # 4. [추가] 등락률 (%) - AI의 '핵심 타겟' 및 '시퀀스' 피처
    df['등락률'] = df['종가'].pct_change() * 100
    df['등락률'] = df['등락률'].fillna(0)
    
    return df
    
# utils.py 의 get_tickers 함수를 아래 내용으로 교체
@st.cache_data
def get_tickers():
    import datetime
    
    # 1. 안전한 날짜 찾기 로직
    # 현재 시각을 기준으로, 장이 열리지 않은 주말이나 새벽이라면 전날 데이터를 찾음
    now = datetime.datetime.now()
    
    # 주말(토:5, 일:6)이면 금요일로 이동
    if now.weekday() == 5: # 토요일
        target_dt = now - datetime.timedelta(days=1)
    elif now.weekday() == 6: # 일요일
        target_dt = now - datetime.timedelta(days=2)
    else:
        # 평일이라도 오전 9시 전이면 전날 데이터를 가져옴
        if now.hour < 9:
            target_dt = now - datetime.timedelta(days=1)
        else:
            target_dt = now
            
    target_date = target_dt.strftime("%Y%m%d")

    try:
        # 2. 직접 계산한 안전한 날짜로 호출 (IndexError 방지)
        tickers = stock.get_market_ticker_list(target_date, market="ALL") # 코스피 기준
        
        # 만약 해당 날짜 데이터가 없다면 10일치 루프를 돌며 가장 최근 데이터를 찾음
        if not tickers:
            for i in range(1, 11):
                past_date = (target_dt - datetime.timedelta(days=i)).strftime("%Y%m%d")
                tickers = stock.get_market_ticker_list(past_date, market="ALL")
                if tickers: break

        return {stock.get_market_ticker_name(t): t for t in tickers}
        
    except Exception as e:
        # 어떤 에러가 나더라도 앱이 죽지 않게 삼성전자 하나라도 반환하는 최후의 보루
        return {"삼성전자": "005930"}
# utils.py 의 변동성 함수 업그레이드 (Winsorizing & Dynamic Cap)
def get_volatility_models(prices):
    # 1. 가격 예외 처리 (0 이하 가격 방지)
    prices = np.where(prices <= 0, 1e-9, prices)
    
    # 2. 로그 수익률 계산 (단위: %)
    returns = np.diff(np.log(prices)) * 100
    
    # [업그레이드 1] 극단치 및 무한대 처리 (윈저라이징)
    # 한국 증시 상하한가를 감안하여 inf는 ±30%로 치환, NaN은 0으로 처리
    returns = np.nan_to_num(returns, nan=0.0, posinf=30.0, neginf=-30.0)
    # 비정상적인 액면분할/오류 데이터 스파이크를 ±35%로 제한 (AI 충격 보존)
    returns = np.clip(returns, -35.0, 35.0) 
    
    train_data = returns[-500:] if len(returns) > 500 else returns 
    
    # [업그레이드 2] 해당 종목의 평소 변동성(Baseline) 계산
    baseline_vol = np.std(train_data)
    if baseline_vol == 0: baseline_vol = 1e-9
    
    # 동적 상한선: 평소 변동성의 10배까지만 허용 (종목마다 다른 기준 적용)
    max_vol_limit = baseline_vol * 10.0 
    
    try:
        # [업그레이드 3] rescale=True 적용으로 내부 알고리즘 최적화 및 발산 방지
        # EGARCH 모델
        egarch_m = arch_model(train_data, vol='EGARCH', p=1, o=1, q=1, dist='t', rescale=True)
        egarch_res = egarch_m.fit(disp='off', show_warning=False)
        egarch_var = egarch_res.forecast(horizon=1).variance.iloc[-1].values[0]
        egarch_vol = np.sqrt(egarch_var) if egarch_var > 0 else baseline_vol
        
        # GJR-GARCH 모델
        gjr_m = arch_model(train_data, p=1, o=1, q=1, dist='t', rescale=True)
        gjr_res = gjr_m.fit(disp='off', show_warning=False)
        gjr_var = gjr_res.forecast(horizon=1).variance.iloc[-1].values[0]
        gjr_vol = np.sqrt(gjr_var) if gjr_var > 0 else baseline_vol
        
    except Exception as e:
        # GARCH 계산 실패 시 평소 변동성으로 대체
        egarch_vol = baseline_vol
        gjr_vol = baseline_vol
        
    # [업그레이드 4] 무식한 30% 커팅 대신, 종목별 맞춤 한계치(max_vol_limit) 적용
    return {
        "egarch": float(np.clip(egarch_vol, 0, max_vol_limit)),
        "gjr_garch": float(np.clip(gjr_vol, 0, max_vol_limit))
    }
