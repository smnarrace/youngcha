import streamlit as st
import pandas as pd
import numpy as np # [추가] 배열 처리를 위해 필요
from pykrx import stock
import datetime

# 필요한 계산 함수들을 utils(또는 logic_calculate)에서 가져옵니다.
from utils import calculate_indicators, get_numerical_analysis, get_volatility_models
from sidebar import render_sidebar
from charts import draw_chart
from results import render_results
from model import YoungChaHybridModel # [추가] 우리가 만든 하이브리드 모델 클래스

# 페이지 설정
st.set_page_config(page_title="AlphaQuant Pro - 검증 시스템", layout="wide")

# --- [NEW] 세션에 하이브리드 모델 장착 ---
# 앱이 재실행되어도 모델의 학습 상태가 초기화되지 않도록 세션에 저장합니다.
if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = YoungChaHybridModel()

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

# --- [NEW] 사이드바에 AI 학습 UI 추가 ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 하이브리드 모델")
if st.session_state.hybrid_model.is_trained:
    st.sidebar.success("✅ 모델 학습 완료 상태입니다.")
else:
    st.sidebar.info("💡 예측 전 먼저 모델을 학습시켜주세요.")

# 2. 데이터 로드
end_date_str = config['target_date'].strftime("%Y%m%d")
# GARCH 500일 윈도우를 위해 충분한 과거 데이터 확보 (약 750일)
start_date_str = (config['target_date'] - datetime.timedelta(days=750)).strftime("%Y%m%d")
df = stock.get_market_ohlcv(start_date_str, end_date_str, config['ticker'])

if not df.empty:
    df = calculate_indicators(df)
    
    # --- [NEW] 하이브리드 모델 집중 학습 로직 ---
    # 데이터가 정상 로드된 상태에서만 사이드바에 학습 버튼을 노출합니다.
    if st.sidebar.button("🚀 모델 집중 학습 시작 (최근 50일)"):
        window_size = 60
        train_size = 50
        
        target_date_ts = pd.Timestamp(config['target_date']).normalize()
        if target_date_ts in df.index:
            max_train_idx = df.index.get_loc(target_date_ts)
        else:
            max_train_idx = len(df) - 1

        # 데이터 길이 검증 (최소 60 + 50 = 110일치 데이터 필요)
        if max_train_idx < window_size + train_size:
            st.sidebar.error("데이터가 부족하여 학습할 수 없습니다. (상장 초기 종목)")
        else:
            with st.spinner("과거 데이터를 수집하고 딥러닝 모델을 훈련하는 중입니다... 영차! 💦"):
                X_seq_list, X_static_list, y_list = [], [], []
                progress_bar = st.sidebar.progress(0)
                
                # 기준일(target_idx) 이전 50일 동안의 데이터를 '문제집'으로 활용
                valid_indices = range(max_train_idx - train_size, max_train_idx)
                
                for i, idx in enumerate(valid_indices):
                    # 1) 시계열 데이터 60일치 (LSTM용)
                    seq = df.iloc[idx - window_size : idx]['종가'].values
                    X_seq_list.append(seq)
                    
                    # 2) 정적 데이터: 해당 시점까지의 데이터로 GARCH 및 수치해석 계산
                    past_prices = df.iloc[:idx]['종가'].values
                    num_res = get_numerical_analysis(past_prices)
                    vol_res = get_volatility_models(past_prices)
                    
                    static_feats = [
                        num_res.get('euler', 0),
                        num_res.get('rk4', 0),
                        num_res.get('newton', 0),
                        vol_res.get('egarch', 0),
                        vol_res.get('gjr_garch', 0)
                    ]
                    X_static_list.append(static_feats)
                    
                    # 3) 정답 라벨 (해당 인덱스 날짜의 종가)
                    y_list.append(df.iloc[idx]['종가'])
                    
                    # 프로그레스 바 업데이트
                    progress_bar.progress((i + 1) / len(valid_indices))
                
                # 리스트를 Numpy 배열로 변환 (딥러닝이 좋아하는 실수형 float32로 강제 변환)
                X_seq_train = np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)
                X_static_train = np.array(X_static_list, dtype=np.float32)
                y_train = np.array(y_list, dtype=np.float32)
                # [NEW] 혹시 수치해석이나 GARCH 계산 중 발생한 NaN(결측치)이 있다면 0으로 안전하게 치환
                X_seq_train = np.nan_to_num(X_seq_train)
                X_static_train = np.nan_to_num(X_static_train)
                y_train = np.nan_to_num(y_train)
                # 모델 훈련 실행
                st.session_state.hybrid_model.train(X_seq_train, X_static_train, y_train)
                st.sidebar.success("🎉 학습 완료!")
                st.rerun() # 학습 완료 후 화면을 새로고침하여 '학습 완료 상태' 텍스트 반영

    # --- 핵심: 데이터 계산부 분리 ---
    # 차트와 결과창에서 공통으로 쓸 데이터를 여기서 미리 계산합니다.
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        if target_idx < 10:
            st.error("데이터가 너무 적어 분석을 시작할 수 없습니다.")
            st.stop()
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
            # 결과창에도 동일한 계산 결과 전달 (results.py 연동 완료)
            render_results(df, config, vol_results=vol_results, predictions=predictions)
    else:
        st.error("선택하신 날짜에 해당하는 데이터가 부족합니다.")
else:
    st.error("데이터를 불러올 수 없습니다.")
