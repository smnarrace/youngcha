import streamlit as st
import datetime
import pandas as pd
from pykrx import stock
from utils import calculate_indicators, get_numerical_analysis, get_volatility_models
from sidebar import render_sidebar_inputs, render_sidebar_actions # 분리된 함수 임포트
from charts import draw_chart
from results import render_results
from model import YoungChaHybridModel

st.set_page_config(page_title="영차 AI 연구소", page_icon="📈", layout="wide")

if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = YoungChaHybridModel()

# 1. 설정값 먼저 받기 (중복 호출 에러 원인 제거)
config = render_sidebar_inputs()

# 2. 데이터 로드
end_date_str = config['target_date'].strftime("%Y%m%d")
start_date_str = (config['target_date'] - datetime.timedelta(days=750)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    # 휴장일 처리
    if target_date_ts not in df.index:
        valid_dates = df.index[df.index <= target_date_ts]
        if not valid_dates.empty:
            target_date_ts = valid_dates[-1]
            config['target_date'] = target_date_ts.date()
        else:
            st.error("데이터가 없습니다."); st.stop()

    # 3. 사이드바 하단 버튼들 출력 (여기서 딱 한 번만 호출!)
    render_sidebar_actions(df, target_date_ts, config)

    # 4. 분석 결과 출력
    target_idx = df.index.get_loc(target_date_ts)
    past_prices = df.iloc[:target_idx + 1]['종가'].values
    predictions = get_numerical_analysis(past_prices, h=config['step_size'])
    vol_results = get_volatility_models(past_prices)
    
    st.title(f"📈 {config['selected_name']} ({config['step_size']}일 주기 분석)")
    
    if not st.session_state.hybrid_model.is_trained:
        st.info("💡 모델 학습 전입니다. 왼쪽 하단의 '학습 시작' 버튼을 눌러주세요.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(draw_chart(df, config, vol_results, predictions), use_container_width=True)
    with col2:
        render_results(df, config, vol_results=vol_results, predictions=predictions)
