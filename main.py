import streamlit as st
from pykrx import stock
import datetime
from utils import calculate_indicators
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
start_date_str = (config['target_date'] - datetime.timedelta(days=250)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    
    st.title(f"📈 {config['selected_name']} ({config['step_size']}일 주기 분석)")
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = draw_chart(df, config)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        render_results(df, config)
else:
    st.error("데이터를 불러올 수 없습니다.")