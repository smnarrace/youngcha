import streamlit as st
import datetime
import pandas as pd
from utils import get_tickers

def render_sidebar():
    with st.sidebar:
        st.title("🔬 영차 AI 전략 조합실")
        st.caption("Numerical + Deep Learning Hybrid System")
        
        # 1. 분석 모드 및 날짜 선택
        app_mode = st.radio("테스트 모드", ["백테스팅"], horizontal=True)
        today = datetime.date.today()
        target_date = st.date_input("📅 분석 기준일 선택", today)

        st.write("---")
        
        # 2. 종목 검색 및 선택
        search_word = st.text_input("🔍 종목명 입력", "삼성전자").strip().lower()
        ticker_dict = get_tickers()
        matched_names = [name for name in ticker_dict.keys() if search_word in name.lower()]
        
        if matched_names:
            selected_name = st.selectbox("🎯 검색 결과 중 선택", matched_names)
            ticker = ticker_dict[selected_name]
        else:
            st.error("❌ 일치하는 종목이 없습니다.")
            st.stop()

        # 3. 예측 주기 설정 (이 값이 모델 학습의 기준이 됨)
        step_size = st.radio("⏱️ 예측 주기 설정", [1, 5], 
                             format_func=lambda x: f"{x}일 기준 예측")
        
        st.write("---")
        
        # 4. AI 하이브리드 엔진 상태 표시
        st.subheader("🤖 AI 하이브리드 엔진")
        if 'hybrid_model' in st.session_state and st.session_state.hybrid_model.is_trained:
            st.success("✅ 엔진 가동 준비 완료")
        else:
            st.info("💡 '모델 집중 학습'을 진행해 주세요.")

        # 내부적으로 필요한 모델 활성화 상태는 딕셔너리로 유지하여 코드 호환성 보장
        models = {"hybrid": True, "rk4": False, "newton": False, "euler": False, "simpson": False}
        vol_models = {"egarch": False, "gjr_garch": False}

    return {
        "target_date": target_date, 
        "selected_name": selected_name, 
        "ticker": ticker,
        "step_size": step_size, 
        "models": models, 
        "vol_models": vol_models,
        "show_bb": False, 
        "show_rsi": False, 
        "ma_settings": [], 
        "show_signals": False
    }
