import streamlit as st
import pandas as pd
import numpy as np
from pykrx import stock
import datetime
import random 

from utils import calculate_indicators, get_numerical_analysis, get_volatility_models
from sidebar import render_sidebar
from charts import draw_chart
from results import render_results
from model import YoungChaHybridModel

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
    df['등락률'] = df['종가'].pct_change() * 100
    df['거래량_변동률'] = df['거래량'].pct_change() * 100
    df = df.dropna()

    def to_pct(val, base):
        return ((val - base) / base) * 100 if base != 0 else 0

    # --- 버튼 1: 모델 새로 학습 ---
    if st.sidebar.button("🚀 모델 전천후 학습 시작"):
        window_size = 60
        h = config['step_size'] # 주기 설정 가져오기
        target_date_ts = pd.Timestamp(config['target_date']).normalize()
        max_idx = df.index.get_loc(target_date_ts) if target_date_ts in df.index else len(df) - 1

        # 데이터 다각화 샘플링
        recent_indices = list(range(max_idx - 30, max_idx))
        year_ago_idx = max_idx - 250
        hist_indices = list(range(year_ago_idx - 20, year_ago_idx)) if year_ago_idx > 120 else []
        
        valid_indices = hist_indices + recent_indices
        random.shuffle(valid_indices)

        if len(valid_indices) < 20:
            st.sidebar.error("데이터가 부족합니다.")
        else:
            with st.spinner(f"{h}일 주기 전천후 학습 중... 영차! 💦"):
                X_seq_list, X_static_list, y_list = [], [], []
                progress_bar = st.sidebar.progress(0)
                
                # [수정] 중복된 루프를 하나로 통합
                for i, idx in enumerate(valid_indices):
                    if idx + h > len(df): continue
                    
                    # 1) LSTM용 시퀀스
                    seq = df.iloc[idx - window_size : idx]['등락률'].values
                    X_seq_list.append(seq)
                    
                    # 2) 정적 데이터 (h 전달)
                    curr_p = df.iloc[idx - 1]['종가']
                    num_res = get_numerical_analysis(df.iloc[:idx]['종가'].values, h=h)
                    vol_res = get_volatility_models(df.iloc[:idx]['종가'].values)
                    
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
                    
                    # 3) h일 뒤의 누적 등락률을 정답으로 설정
                    future_p = df.iloc[idx + h - 1]['종가']
                    y_list.append(to_pct(future_p, curr_p))
                    
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
            with st.spinner("검증 데이터 생성 중..."):
                st.session_state.history = []
                h = config['step_size']
                check_days = 30
                target_date_ts = pd.Timestamp(config['target_date']).normalize()
                val_idx = df.index.get_loc(target_date_ts) if target_date_ts in df.index else len(df) - 1
                
                # [수정] 검증 시에도 학습과 동일한 h와 피처를 사용
                for i in range(val_idx - check_days, val_idx + 1):
                    if i < 60 or i + h > len(df): continue
                    
                    curr_p = df.iloc[i - 1]['종가']
                    test_seq = df.iloc[i - 60 : i]['등락률'].values
                    num_res = get_numerical_analysis(df.iloc[:i]['종가'].values, h=h)
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
                    
                    actual_future_p = df.iloc[i + h - 1]['종가']
                    pred_future_p = curr_p * (1 + pred_pct / 100)
                    
                    st.session_state.history.append({
                        "date": df.index[i].date(), 
                        "actual": actual_future_p, 
                        "pred": pred_future_p,
                        "hit": (pred_future_p > curr_p and actual_future_p > curr_p) or \
                               (pred_future_p < curr_p and actual_future_p < curr_p)
                    })
                st.sidebar.success("📈 검증 완료!")
                st.rerun()

    # 결과 출력 (이전과 동일)
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
