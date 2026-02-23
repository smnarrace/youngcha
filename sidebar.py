import streamlit as st
import datetime
from utils import get_tickers

def render_sidebar():
    with st.sidebar:
        st.title("🔬 전략 조합실")
        app_mode = st.radio("🏠 분석 모드 선택", ["현재 예측", "백테스팅"], horizontal=True)
        
        target_date = datetime.date.today()
        if app_mode == "백테스팅":
            target_date = st.date_input("📅 분석 기준일 선택", datetime.date.today())
        else:
            st.caption("💡 '오늘 날짜 기준 분석' ")

        st.write("---")
        search_word = st.text_input("🔍 종목명 입력", "삼성전자").strip().lower()
        ticker_dict = get_tickers()
        matched_names = [name for name in ticker_dict.keys() if search_word in name.lower()]
        
        if matched_names:
            selected_name = st.selectbox("🎯 검색 결과 중 선택", matched_names)
            ticker = ticker_dict[selected_name]
        else:
            st.error("❌ 일치하는 종목이 없습니다.")
            st.stop()

        step_size = st.radio("⏱️ 예측 주기 설정", [1, 5], format_func=lambda x: f"{x}일 기준")
        st.write("---")

        st.subheader(" 🧪 수치해석 모델")
        c1, c2 = st.columns(2)
        models = {
            "rk4": c1.checkbox("RK4", value=False),
            "newton": c1.checkbox("뉴턴", value=False),
            "euler": c2.checkbox("오일러", value=False),
            "simpson": c2.checkbox("심슨", value=False)
        }

        st.subheader(" 📊 보조지표")
        c3, c4 = st.columns(2)
        show_bb = c3.checkbox("BB", value=False)
        show_rsi = c4.checkbox("RSI", value=False)
        use_ma = c3.checkbox("MA", value=False)
        use_ema = c4.checkbox("EMA", value=False)
        
        ma_settings = []
        if use_ma or use_ema:
            st.caption("⚙️ 이동평균선 세부 설정")
            color_options = {"🔴 Red": "#FF4B4B", "🔵 Blue": "#1C83E1", "🟢 Green": "#00C805", 
                             "🟡 Yellow": "#FBC02D", "🟣 Purple": "#AB47BC", "⚪ White": "#FFFFFF", "❌ 색상 없음": None}

            for i in range(1, 4):
                st.markdown(f"**Line #{i}**")
                sc1, sc2, sc3, sc4 = st.columns([1.2, 1, 1.5, 1])
                m_type = sc1.selectbox(f"유형_{i}", ["MA", "EMA"], key=f"ma_type_{i}", label_visibility="collapsed")
                period = sc2.number_input(f"기간_{i}", 1, 200, [5, 10, 20][i-1], key=f"ma_p_{i}", label_visibility="collapsed")
                color_name = sc3.selectbox(f"색상_{i}", list(color_options.keys()), index=i-1, key=f"ma_c_name_{i}", label_visibility="collapsed")
                chosen_color = color_options[color_name]
                width = sc4.slider(f"굵기_{i}", 1, 5, 2, key=f"ma_w_{i}", label_visibility="collapsed")
                
                if chosen_color:
                    ma_settings.append({"active": True, "type": m_type, "period": period, "color": chosen_color, "width": width})

        st.subheader("🖼️ 시각화 옵션")
        show_signals = st.checkbox("차트 시그널(원) 표시", value=False)
        
    return {
        "target_date": target_date, "selected_name": selected_name, "ticker": ticker,
        "step_size": step_size, "models": models, "show_bb": show_bb, 
        "show_rsi": show_rsi, "ma_settings": ma_settings, "show_signals": show_signals
    }