import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random # [추가] 데이터 셔플용

from utils import calculate_indicators, get_numerical_analysis, get_volatility_models
from sidebar import render_sidebar
from charts import draw_chart
from results import render_results
from model import YoungChaHybridModel

# 페이지 설정
st.set_page_config(page_title="AlphaQuant Pro - 전천후 하이브리드", layout="wide")

if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = YoungChaHybridModel()

config = render_sidebar()

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 하이브리드 엔진 (V2)")
if st.session_state.hybrid_model.is_trained:
    st.sidebar.success("✅ 전천후 모델 학습 완료")
else:
    st.sidebar.info("💡 모델을 학습시켜 '시장의 본질'을 가르쳐주세요.")

# 데이터 로드
end_date_str = config['target_date'].strftime("%Y%m%d")
start_date_str = (config['target_date'] - datetime.timedelta(days=750)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    
    # [전략 2 기초] 거래량 변동률 등 추가 지표 생성
    df['등락률'] = df['종가'].pct_change() * 100
    df['거래량_변동률'] = df['거래량'].pct_change() * 100
    df = df.dropna()

    # --- [전략 1] 수치 모델 % 변환 함수 ---
    def to_pct(val, base):
        return ((val - base) / base) * 100 if base != 0 else 0

    # --- 버튼 1: 모델 새로 학습 (전략 3 적용) ---
    if st.sidebar.button("🚀 모델 전천후 학습 시작"):
        window_size = 60
        target_date_ts = pd.Timestamp(config['target_date']).normalize()
        max_idx = df.index.get_loc(target_date_ts) if target_date_ts in df.index else len(df) - 1

        # [전략 3] 학습 데이터 다각화 (최근 30일 + 1년 전 20일)
        recent_indices = list(range(max_idx - 30, max_idx))
        year_ago_idx = max_idx - 250 # 약 1년 전 거래일 시점
        hist_indices = list(range(year_ago_idx - 20, year_ago_idx)) if year_ago_idx > 100 else []
        
        valid_indices = hist_indices + recent_indices
        random.shuffle(valid_indices) # 무작위로 섞어 편향 제거

        if len(valid_indices) < 20:
            st.sidebar.error("학습을 위한 데이터 시계열이 부족합니다.")
        else:
            with st.spinner("상승장과 하락장을 모두 가르치는 중... 영차! 💦"):
                X_seq_list, X_static_list, y_list = [], [], []
                progress_bar = st.sidebar.progress(0)
                
                for i, idx in enumerate(valid_indices):
                    # 1) LSTM용: 등락률 시퀀스
                    seq = df.iloc[idx - window_size : idx]['등락률'].values
                    X_seq_list.append(seq)
                    
                    # 2) [전략 1 & 2] 정적 데이터 가공
                    curr_p = df.iloc[idx - 1]['종가']
                    num_res = get_numerical_analysis(df.iloc[:idx]['종가'].values)
                    vol_res = get_volatility_models(df.iloc[:idx]['종가'].values)
                    
                    # 수치 모델 % 변환 + RSI/거래량 추가
                    static_feats = [
                        to_pct(num_res.get('euler', curr_p), curr_p),
                        to_pct(num_res.get('rk4', curr_p), curr_p),
                        to_pct(num_res.get('newton', curr_p), curr_p),
                        vol_res.get('egarch', 0),
                        vol_res.get('gjr_garch', 0),
                        df.iloc[idx - 1].get('RSI', 50),
                        df.iloc[idx - 1].get('거래량_변동률', 0)
                    ]
                    X_static_list.append(static_feats)
                    y_list.append(df.iloc[idx]['등락률'])
                    progress_bar.progress((i + 1) / len(valid_indices))
                
                st.session_state.hybrid_model.train(
                    np.nan_to_num(np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)),
                    np.nan_to_num(np.array(X_static_list, dtype=np.float32)),
                    np.nan_to_num(np.array(y_list, dtype=np.float32))
                )
                st.sidebar.success("🎉 전천후 학습 완료!")
                st.rerun()

    # --- 버튼 2: 현재 모델로 이 구간 검증 ---
    if st.sidebar.button("🔍 현재 모델로 이 구간 검증 (15일)"):
        if not st.session_state.hybrid_model.is_trained:
            st.sidebar.error("먼저 모델을 학습시켜주세요!")
        else:
            with st.spinner(f"{config['target_date']} 기준 검증 중..."):
                st.session_state.history = []
                target_date_ts = pd.Timestamp(config['target_date']).normalize()
                val_idx = df.index.get_loc(target_date_ts) if target_date_ts in df.index else len(df) - 1
                
                for i in range(val_idx - 15, val_idx + 1):
                    if i < 60: continue
                    curr_p = df.iloc[i - 1]['종가']
                    
                    test_seq = df.iloc[i - 60 : i]['등락률'].values
                    num_res = get_numerical_analysis(df.iloc[:i]['종가'].values)
                    vol_res = get_volatility_models(df.iloc[:i]['종가'].values)
                    
                    X_static_test = np.array([[
                        to_pct(num_res.get('euler', curr_p), curr_p),
                        to_pct(num_res.get('rk4', curr_p), curr_p),
                        to_pct(num_res.get('newton', curr_p), curr_p),
                        vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0),
                        df.iloc[i - 1].get('RSI', 50),
                        df.iloc[i - 1].get('거래량_변동률', 0)
                    ]], dtype=np.float32)
                    
                    pred_pct = st.session_state.hybrid_model.predict(
                        np.nan_to_num(test_seq.reshape(1, 60, 1)), 
                        np.nan_to_num(X_static_test)
                    )[0]
                    
                    actual_p = df.iloc[i]['종가']
                    pred_p = curr_p * (1 + pred_pct / 100)
                    st.session_state.history.append({
                        "date": df.index[i].date(), "actual": actual_p, "pred": pred_p,
                        "hit": (pred_p > curr_p and actual_p > curr_p) or (pred_p < curr_p and actual_p < curr_p)
                    })
                st.sidebar.success("📈 검증 완료!")
                st.rerun()

    # 결과 출력부
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        past_prices = df.iloc[:target_idx + 1]['종가'].values
        predictions = get_numerical_analysis(past_prices, h=config['step_size'])
        vol_results = get_volatility_models(past_prices)
        
        st.title(f"📈 {config['selected_name']} ({config['step_size']}일 주기 분석)")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(draw_chart(df, config, vol_results, predictions), use_container_width=True)
        with col2:
            render_results(df, config, vol_results=vol_results, predictions=predictions)
