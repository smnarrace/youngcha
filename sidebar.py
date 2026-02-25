import streamlit as st
import datetime
import pandas as pd
import numpy as np
import random
from utils import get_tickers, get_numerical_analysis, get_volatility_models

def render_sidebar_inputs():
    with st.sidebar:
        st.title("🔬 영차 AI 전략 조합실")
        st.caption("Numerical + Deep Learning Hybrid System")
        
        app_mode = st.radio("테스트 모드", ["백테스팅"], horizontal=True)
        today = datetime.date.today()
        target_date = st.date_input("📅 분석 기준일 선택", today)

        st.write("---")
        
        search_word = st.text_input("🔍 종목명 입력", "삼성전자").strip().lower()
        ticker_dict = get_tickers()
        matched_names = [name for name in ticker_dict.keys() if search_word in name.lower()]
        
        if matched_names:
            selected_name = st.selectbox("🎯 검색 결과 중 선택", matched_names)
            ticker = ticker_dict[selected_name]
        elif search_word and search_word != "삼성전자":
            st.error("❌ 일치하는 종목이 없습니다.")
            selected_name, ticker = "삼성전자", "005930"
        else:
            selected_name, ticker = "삼성전자", "005930"

        step_size = st.radio("⏱️ 예측 주기 설정", [1, 5], format_func=lambda x: f"{x}일 기준 예측")
        
        models = {"hybrid": True, "rk4": False, "newton": False, "euler": False, "simpson": False}
        vol_models = {"egarch": False, "gjr_garch": False}
        
        return {
            "target_date": target_date, "selected_name": selected_name, 
            "ticker": ticker, "step_size": step_size,
            "models": models, "vol_models": vol_models,
            "show_bb": False, "show_rsi": False, "ma_settings": [], "show_signals": False
        }

def render_sidebar_actions(df, target_date_ts, config):
    with st.sidebar:
        st.write("---")
        st.subheader("🤖 AI 하이브리드 엔진 (Residual v2)")
        
        if st.session_state.hybrid_model.is_trained:
            st.success("✅ 전천후 잔차 모델 학습 완료")
        else:
            st.info("💡 모델을 학습시켜주세요.")

        def to_pct(val, base):
            return ((val - base) / base) * 100 if base != 0 else 0

        # --- 버튼 1: 모델 학습 (3대 모델 잔차 로직) ---
        if st.button("🚀 모델 전천후 학습 시작"):
            window_size, h = 60, config['step_size']
            max_idx = df.index.get_loc(target_date_ts)
            
            recent_indices = list(range(max_idx - 100, max_idx))
            year_ago_idx = max_idx - 250
            hist_indices = list(range(year_ago_idx - 50, year_ago_idx)) if year_ago_idx > 150 else []
            valid_indices = hist_indices + recent_indices
            random.shuffle(valid_indices)

            if len(valid_indices) < 20:
                st.error("데이터 부족")
            else:
                with st.spinner(f"{len(valid_indices)}개 샘플로 잔차 학습 중..."):
                    X_seq_list, X_static_list = [], []
                    y_actual_list, y_base_list = [], [] # 잔차 계산을 위한 두 리스트

                    for idx in valid_indices:
                        if idx + h >= len(df) or idx < window_size: continue
                        
                        curr_p = df.iloc[idx - 1]['종가']
                        num_res = get_numerical_analysis(df.iloc[:idx]['종가'].values, h=h)
                        vol_res = get_volatility_models(df.iloc[:idx]['종가'].values)

                        # [핵심] 3대 수치 모델 평균 베이스라인 계산
                        euler_p = to_pct(num_res.get('euler', curr_p), curr_p)
                        rk4_p = to_pct(num_res.get('rk4', curr_p), curr_p)
                        newton_p = to_pct(num_res.get('newton', curr_p), curr_p)
                        base_p = (euler_p + rk4_p + newton_p) / 3

                        # 실제 등락률
                        actual_p = to_pct(df.iloc[idx + h - 1]['종가'], curr_p)

                        static_feats = [
                            base_p, # 평균 베이스라인 자체도 중요한 정보
                            euler_p, rk4_p, newton_p,
                            vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0),
                            df.iloc[idx - 1].get('RSI', 50),
                            df.iloc[idx - 1].get('거래량_변동률', 0)
                        ]
                        
                        X_seq_list.append(df.iloc[idx - window_size : idx]['등락률'].values)
                        X_static_list.append(static_feats)
                        y_actual_list.append(actual_p)
                        y_base_list.append(base_p)
                    
                    # 모델 학습 호출 (실제값과 수치해석 베이스를 따로 넘김)
                    st.session_state.hybrid_model.train(
                        np.nan_to_num(np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)),
                        np.nan_to_num(np.array(X_static_list, dtype=np.float32)),
                        np.array(y_actual_list), 
                        np.array(y_base_list)
                    )
                    st.success("잔차 학습 완료!"); st.rerun()

        # --- 버튼 2: 검증 (잔차 반영 버전) ---
        if st.button("🔍 현재 모델로 이 구간 검증 (30일)"):
            if not st.session_state.hybrid_model.is_trained:
                st.error("모델 학습 필요")
            else:
                with st.spinner("잔차 보정 검증 중..."):
                    st.session_state.history = []
                    h, val_idx = config['step_size'], df.index.get_loc(target_date_ts)
                    for i in range(val_idx - 30, val_idx + 1):
                        if i < 60 or i + h >= len(df): continue
                        curr_p = df.iloc[i - 1]['종가']
                        num_res = get_numerical_analysis(df.iloc[:i]['종가'].values, h=h)
                        vol_res = get_volatility_models(df.iloc[:i]['종가'].values)

                        # 수치해석 평균 베이스라인
                        base_p = (to_pct(num_res.get('euler', curr_p), curr_p) + 
                                  to_pct(num_res.get('rk4', curr_p), curr_p) + 
                                  to_pct(num_res.get('newton', curr_p), curr_p)) / 3

                        X_static_test = np.array([[
                            base_p, to_pct(num_res.get('euler', curr_p), curr_p), to_pct(num_res.get('rk4', curr_p), curr_p),
                            to_pct(num_res.get('newton', curr_p), curr_p), vol_res.get('egarch', 0), 
                            vol_res.get('gjr_garch', 0), df.iloc[i - 1].get('RSI', 50), df.iloc[i - 1].get('거래량_변동률', 0)
                        ]], dtype=np.float32)
                        
                        # AI의 보정값(Residual) 예측
                        ai_residual = st.session_state.hybrid_model.predict(
                            np.nan_to_num(df.iloc[i - 60 : i]['등락률'].values.reshape(1, 60, 1)), 
                            np.nan_to_num(X_static_test)
                        )[0]

                        # 최종 예측 = 수학적 평균 + AI 보정
                        final_pred_pct = base_p + ai_residual
                        
                        actual_p = to_pct(df.iloc[i + h - 1]['종가'], curr_p)
                        st.session_state.history.append({
                            "date": df.index[i].date(), "actual": df.iloc[i + h - 1]['종가'], 
                            "pred": curr_p * (1 + final_pred_pct / 100),
                            "hit": (final_pred_pct > 0 and actual_p > 0) or (final_pred_pct < 0 and actual_p < 0)
                        })
                    st.success("검증 완료"); st.rerun()
