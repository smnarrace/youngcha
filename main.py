import streamlit as st
import pandas as pd
import numpy as np
from pykrx import stock
import datetime

# 필요한 계산 함수들 가져오기
from utils import calculate_indicators, get_numerical_analysis, get_volatility_models
from sidebar import render_sidebar
from charts import draw_chart
from results import render_results
from model import YoungChaHybridModel

# 페이지 설정
st.set_page_config(page_title="AlphaQuant Pro - 하이브리드 검증", layout="wide")

if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = YoungChaHybridModel()

# 1. 사이드바 렌더링
config = render_sidebar()

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 하이브리드 엔진")
if st.session_state.hybrid_model.is_trained:
    st.sidebar.success("✅ 모델 학습 완료 상태")
else:
    st.sidebar.info("💡 먼저 모델을 학습시켜주세요.")

# 2. 데이터 로드
end_date_str = config['target_date'].strftime("%Y%m%d")
start_date_str = (config['target_date'] - datetime.timedelta(days=750)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    
    # [중요] 등락률 데이터 생성 (모든 학습/예측의 기준)
    df['등락률'] = df['종가'].pct_change() * 100
    df = df.dropna()

    # --- 버튼 1: 모델 새로 학습 ---
    if st.sidebar.button("🚀 모델 집중 학습 시작 (최근 50일)"):
        window_size = 60
        train_size = 50
        target_date_ts = pd.Timestamp(config['target_date']).normalize()
        max_train_idx = df.index.get_loc(target_date_ts) if target_date_ts in df.index else len(df) - 1

        if max_train_idx < window_size + train_size:
            st.sidebar.error("데이터가 부족합니다.")
        else:
            with st.spinner("AI가 등락률 패턴을 공부 중입니다... 영차! 💦"):
                X_seq_list, X_static_list, y_list = [], [], []
                progress_bar = st.sidebar.progress(0)
                
                valid_indices = range(max_train_idx - train_size, max_train_idx)
                for i, idx in enumerate(valid_indices):
                    # [변경] 종가 대신 '등락률' 시퀀스 추출
                    seq = df.iloc[idx - window_size : idx]['등락률'].values
                    X_seq_list.append(seq)
                    
                    past_prices = df.iloc[:idx]['종가'].values
                    num_res = get_numerical_analysis(past_prices)
                    vol_res = get_volatility_models(past_prices)
                    
                    static_feats = [
                        num_res.get('euler', 0), num_res.get('rk4', 0), num_res.get('newton', 0),
                        vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0)
                    ]
                    X_static_list.append(static_feats)
                    y_list.append(df.iloc[idx]['등락률']) # 정답도 등락률
                    progress_bar.progress((i + 1) / len(valid_indices))
                
                X_seq_train = np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)
                X_static_train = np.array(X_static_list, dtype=np.float32)
                y_train = np.array(y_list, dtype=np.float32)
                
                st.session_state.hybrid_model.train(np.nan_to_num(X_seq_train), 
                                                    np.nan_to_num(X_static_train), 
                                                    np.nan_to_num(y_train))
                st.sidebar.success("🎉 학습 완료!")
                st.rerun()

    # --- [★NEW] 버튼 2: 현재 모델로 이 구간 검증 (재학습 없음) ---
    if st.sidebar.button("🔍 현재 모델로 이 구간 검증 (15일)"):
        if not st.session_state.hybrid_model.is_trained:
            st.sidebar.error("먼저 모델을 학습시켜야 검증이 가능합니다!")
        else:
            with st.spinner(f"오늘의 뇌로 {config['target_date']} 기준 과거 15일을 채점 중..."):
                st.session_state.history = []
                check_days = 15
                window_size = 60
                
                target_date_ts = pd.Timestamp(config['target_date']).normalize()
                val_idx = df.index.get_loc(target_date_ts) if target_date_ts in df.index else len(df) - 1

                for i in range(val_idx - check_days + 1, val_idx + 1):
                    if i < window_size: continue
                    
                    actual_p = df.iloc[i]['종가']
                    prev_p = df.iloc[i - 1]['종가']
                    
                    # 1) 입력 데이터 준비 (등락률 기반)
                    test_seq = df.iloc[i - window_size : i]['등락률'].values
                    X_seq_test = np.array(test_seq, dtype=np.float32).reshape(1, window_size, 1)
                    
                    past_prices_test = df.iloc[:i]['종가'].values
                    test_num = get_numerical_analysis(past_prices_test)
                    test_vol = get_volatility_models(past_prices_test)
                    
                    X_static_test = np.array([[
                        test_num.get('euler', 0), test_num.get('rk4', 0),
                        test_num.get('newton', 0), test_vol.get('egarch', 0),
                        test_vol.get('gjr_garch', 0)
                    ]], dtype=np.float32)
                    
                    # 2) 예측 (등락률 예측 -> 가격 복구)
                    pred_pct = st.session_state.hybrid_model.predict(np.nan_to_num(X_seq_test), 
                                                                     np.nan_to_num(X_static_test))[0]
                    pred_p = prev_p * (1 + pred_pct / 100)
                    
                    # 3) 히스토리 저장
                    st.session_state.history.append({
                        "date": df.index[i].date(), "actual": actual_p, "pred": pred_p,
                        "hit": (pred_p > prev_p and actual_p > prev_p) or (pred_p < prev_p and actual_p < prev_p)
                    })
                
                st.sidebar.success(f"📈 {check_days}일간의 과거 대입 검증 완료!")
                st.rerun()

    # --- 4. 메인 분석 결과 출력 ---
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
            render_results(df, config, vol_results, predictions)
    else:
        st.error("데이터 부족")
else:
    st.error("데이터 로드 실패")
