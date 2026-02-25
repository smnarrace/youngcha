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

st.set_page_config(page_title="AlphaQuant Pro - 영차 AI 연구소", layout="wide")

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
    
    # 🎯 [1단계] 휴장일 자동 조정 로직 (9월 연휴 에러 방지)
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    if target_date_ts not in df.index:
        valid_dates = df.index[df.index <= target_date_ts]
        if not valid_dates.empty:
            target_date_ts = valid_dates[-1]
            st.sidebar.warning(f"💡 선택하신 날은 휴장일입니다. \n가장 최근 영업일인 {target_date_ts.date()} 기준으로 분석합니다.")
        else:
            st.error("❌ 조회 범위 내에 유효한 영업일 데이터가 없습니다.")
            st.stop()

    def to_pct(val, base):
        return ((val - base) / base) * 100 if base != 0 else 0

    # --- 버튼 1: 모델 새로 학습 (대량 샘플링 전략) ---
    if st.sidebar.button("🚀 모델 전천후 학습 시작"):
        window_size = 60
        h = config['step_size']
        max_idx = df.index.get_loc(target_date_ts)

        # [전략 3] 학습 샘플 대폭 확장 (최근 100일 + 1년전 50일 + 2년전 50일)
        recent_indices = list(range(max_idx - 100, max_idx))
        year_ago_idx = max_idx - 250
        two_years_idx = max_idx - 500
        
        hist_1 = list(range(year_ago_idx - 50, year_ago_idx)) if year_ago_idx > 150 else []
        hist_2 = list(range(two_years_idx - 50, two_years_idx)) if two_years_idx > 150 else []
        
        valid_indices = hist_1 + hist_2 + recent_indices
        random.shuffle(valid_indices)

        if len(valid_indices) < 30:
            st.sidebar.error("데이터가 부족합니다.")
        else:
            with st.spinner(f"샘플 {len(valid_indices)}개로 {h}일 주기 전천후 학습 중..."):
                X_seq_list, X_static_list, y_list = [], [], []
                
                for idx in valid_indices:
                    if idx + h > len(df) or idx < window_size: continue
                    
                    # 1) 피처 생성 (% 변환 적용)
                    seq = df.iloc[idx - window_size : idx]['등락률'].values
                    curr_p = df.iloc[idx - 1]['종가']
                    num_res = get_numerical_analysis(df.iloc[:idx]['종가'].values, h=h)
                    vol_res = get_volatility_models(df.iloc[:idx]['종가'].values)
                    
                    static_feats = [
                        to_pct(num_res.get('euler', curr_p), curr_p),
                        to_pct(num_res.get('rk4', curr_p), curr_p),
                        to_pct(num_res.get('newton', curr_p), curr_p),
                        vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0),
                        df.iloc[idx - 1].get('RSI', 50),
                        df.iloc[idx - 1].get('거래량_변동률', 0)
                    ]
                    X_seq_list.append(seq)
                    X_static_list.append(static_feats)
                    
                    # 2) 정답 설정 (h일 뒤 누적 등락률)
                    future_p = df.iloc[idx + h - 1]['종가']
                    y_list.append(to_pct(future_p, curr_p))
                
                st.session_state.hybrid_model.train(
                    np.nan_to_num(np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)),
                    np.nan_to_num(np.array(X_static_list, dtype=np.float32)),
                    np.nan_to_num(np.array(y_list, dtype=np.float32))
                )
                st.sidebar.success("🎉 전천후 학습 완료!")
                st.rerun()

    # --- 버튼 2: 현재 모델로 이 구간 검증 (30일) ---
    if st.sidebar.button("🔍 현재 모델로 이 구간 검증 (30일)"):
        if not st.session_state.hybrid_model.is_trained:
            st.sidebar.error("먼저 모델을 학습시켜주세요!")
        else:
            with st.spinner("최근 30일 실력을 채점 중..."):
                st.session_state.history = []
                h = config['step_size']
                val_idx = df.index.get_loc(target_date_ts)
                
                for i in range(val_idx - 30, val_idx + 1):
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

    # 결과 출력부 (조정된 target_date_ts 사용)
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
else:
    st.error("데이터 로드 실패")
        with col1:
            st.plotly_chart(draw_chart(df, config, vol_results, predictions), use_container_width=True)
        with col2:
            render_results(df, config, vol_results=vol_results, predictions=predictions)
