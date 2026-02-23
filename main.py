import streamlit as st
import pandas as pd
from pykrx import stock
import datetime
# 필요한 계산 함수들을 utils(또는 logic_calculate)에서 가져옵니다.
from utils import calculate_indicators, get_numerical_analysis, get_volatility_models
from sidebar import render_sidebar
from charts import draw_chart
from results import render_results

# 페이지 설정
st.set_page_config(page_title="AlphaQuant Pro - 검증 시스템", layout="wide")

# CSS 로드
st.markdown("""
    <style>
        .block-container {padding-top: 1rem !important;}
        [data-testid="stSidebarNav"] {display: none;}
        .stHeadingContainer {margin-top: -2rem !important;}
    </style>
""", unsafe_allow_html=True)

# 1. 사이드바 렌더링 및 설정값 가져오기
config = render_sidebar()

# 2. 데이터 로드
end_date_str = config['target_date'].strftime("%Y%m%d")
# GARCH 500일 윈도우를 위해 충분한 과거 데이터 확보 (약 750일)
start_date_str = (config['target_date'] - datetime.timedelta(days=750)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    
    # --- 핵심: 데이터 계산부 분리 ---
    # 차트와 결과창에서 공통으로 쓸 데이터를 여기서 미리 계산합니다.
    target_date_ts = pd.Timestamp(config['target_date'])
    
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        past_prices = df.iloc[:target_idx + 1]['종가'].values # 기준일 포함 데이터
        
        # 수치해석 및 변동성 모델 계산
        predictions = get_numerical_analysis(past_prices, h=config['step_size'])
        vol_results = get_volatility_models(past_prices)
        
        st.title(f"📈 {config['selected_name']} ({config['step_size']}일 주기 분석)")
        col1, col2 = st.columns([3, 2])

        with col1:
            # 차트에 계산된 predictions와 vol_results를 전달
            fig = draw_chart(df, config, vol_results=vol_results, predictions=predictions)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 결과창에도 동일한 계산 결과 전달 (results.py도 인자를 받게 수정 필요)
            render_results(df, config, vol_results=vol_results, predictions=predictions)
    else:
        st.error("선택하신 날짜에 해당하는 데이터가 부족합니다.")
else:
    st.error("데이터를 불러올 수 없습니다.")
