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

st.set_page_config(
    page_title="수학으로 주가예측 시뮬레이", # 구글 검색 시 제목으로 뜸
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/smnarrace/youngcha',
        'About': "# 영차 AI 연구소\n수치 해석과 딥러닝을 결합한 주가 예측 웹서비스입니다." # 검색 로봇이 읽어갈 설명
    }
)

if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = YoungChaHybridModel()

config = render_sidebar()

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 하이브리드 엔진 (V2)")
if st.session_state.hybrid_model.is_trained:
    st.sidebar.success("✅ 전천후 모델 학습 완료")
else:
    st.sidebar.info("💡 모델을 학습시켜주세요.")

# 데이터 로드
end_date_str = config['target_date'].strftime("%Y%m%d")
start_date_str = (config['target_date'] - datetime.timedelta(days=750)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    
    # 🎯 [핵심 1] 휴장일/주말 방어 로직 & 분석 기준일 동기화
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    if target_date_ts not in df.index:
        valid_dates = df.index[df.index <= target_date_ts]
        if not valid_dates.empty:
            target_date_ts = valid_dates[-1]
            # [중요] config의 날짜 자체를 영업일로 교체해줘야 results/charts에서도 에러가 안 납니다!
            config['target_date'] = target_date_ts.date()
            st.sidebar.warning(f"💡 휴장일입니다. \n최근 영업일인 {target_date_ts.date()} 기준 분석")
        else:
            st.error("데이터가 없습니다.")
            st.stop()

    def to_pct(val, base):
        return ((val - base) / base) * 100 if base != 0 else 0

    # --- 버튼 1: 모델 새로 학습 (중복 루프 버그 수정 버전) ---
    if st.sidebar.button("🚀 모델 전천후 학습 시작"):
        window_size = 60
        h = config['step_size']
        max_idx = df.index.get_loc(target_date_ts)

        # [전략 3] 학습 샘플 대폭 확장
        recent_indices = list(range(max_idx - 100, max_idx))
        year_ago_idx = max_idx - 250
        hist_indices = list(range(year_ago_idx - 50, year_ago_idx)) if year_ago_idx > 150 else []
        
        valid_indices = hist_indices + recent_indices
        random.shuffle(valid_indices)

        if len(valid_indices) < 20:
            st.sidebar.error("데이터 부족")
        else:
            with st.spinner(f"샘플 {len(valid_indices)}개 학습 중..."):
                X_seq_list, X_static_list, y_list = [], [], []
                # [수정] 중복된 2중 for문을 하나로 통합
                for idx in valid_indices:
                    if idx + h >= len(df) or idx < window_size: continue
                    
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
                    y_list.append(to_pct(df.iloc[idx + h - 1]['종가'], curr_p))
                
                st.session_state.hybrid_model.train(
                    np.nan_to_num(np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)),
                    np.nan_to_num(np.array(X_static_list, dtype=np.float32)),
                    np.nan_to_num(np.array(y_list, dtype=np.float32))
                )
                st.sidebar.success("학습 완료!")
                st.rerun()

    # --- 버튼 2: 검증 (30일) ---
    if st.sidebar.button("🔍 현재 모델로 이 구간 검증 (30일)"):
        if not st.session_state.hybrid_model.is_trained:
            st.sidebar.error("모델 학습 필요")
        else:
            with st.spinner("검증 중..."):
                st.session_state.history = []
                h = config['step_size']
                val_idx = df.index.get_loc(target_date_ts)
                for i in range(val_idx - 30, val_idx + 1):
                    if i < 60 or i + h >= len(df): continue
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
                        np.nan_to_num(test_seq.reshape(1, 60, 1)), np.nan_to_num(X_static_test)
                    )[0]
                    st.session_state.history.append({
                        "date": df.index[i].date(), "actual": df.iloc[i + h - 1]['종가'],
                        "pred": curr_p * (1 + pred_pct / 100),
                        "hit": (pred_pct > 0 and (df.iloc[i + h - 1]['종가'] > curr_p)) or (pred_pct < 0 and (df.iloc[i + h - 1]['종가'] < curr_p))
                    })
                st.sidebar.success("검증 완료")
                st.rerun()

    # [핵심 2] 결과 출력 (Indentation Error 해결 & Adjusted Date 적용)
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
