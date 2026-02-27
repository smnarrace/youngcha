import streamlit as st
import datetime
import pandas as pd
import numpy as np
import random
from utils import get_tickers, get_numerical_analysis, get_volatility_models, get_coin_tickers

def render_sidebar_inputs():
    with st.sidebar:
        st.title("🔬 영차 AI 전략 조합실")
        st.caption("Numerical + Deep Learning Hybrid System")
        
        app_mode = st.radio("테스트 모드", ["백테스팅"], horizontal=True)
        today = datetime.date.today()
        target_date = st.date_input("📅 분석 기준일 선택", today)
        
        st.write("---")
        market_type = st.radio("🌐 시장 선택", ["주식 (한국)", "가상화폐 (KRW)"], horizontal=True)
        
        # 🛡️ 방어 전략 설정 UI
        buy_threshold = st.slider(
            "🎯 매수 문턱값 (%)", 1.0, 3.0, 1.0, step=0.1,
            help="AI의 상승 예측치가 이 값보다 높을 때만 매수합니다."
        )
        
        vol_limit = st.slider(
            "⚠️ 변동성 제한 (GARCH)", 3.0, 15.0, 10.0, step=0.5,
            help="시장 변동성(EGARCH)이 이 수치보다 높으면 '위험'으로 간주하고 매수하지 않습니다."
        )

        if market_type == "주식 (한국)":
            search_word = st.text_input("🔍 종목명 입력", "삼성전자").strip().lower()
            ticker_dict = get_tickers()
            default_name, default_ticker = "삼성전자", "005930"
        else:
            search_word = st.text_input("🔍 코인 심볼 입력 (예: BTC)", "BTC").strip().upper()
            ticker_dict = get_coin_tickers()
            default_name, default_ticker = "BTC", "KRW-BTC"
            
        matched_names = [name for name in ticker_dict.keys() if search_word in name.upper() or search_word in name.lower()]
        
        if matched_names:
            selected_name = st.selectbox("🎯 검색 결과 중 선택", matched_names)
            ticker = ticker_dict[selected_name]
        elif search_word:
            st.error("❌ 일치하는 종목/코인이 없습니다.")
            selected_name, ticker = default_name, default_ticker
        else:
            selected_name, ticker = default_name, default_ticker

        step_size = st.radio("⏱️ 예측 주기 설정", [1, 5], format_func=lambda x: f"{x}일 기준 예측")
        
        models = {"hybrid": True, "rk4": False, "newton": False, "euler": False, "simpson": False}
        vol_models = {"egarch": False, "gjr_garch": False}
        
        return {
            "market_type": market_type,
            "target_date": target_date, "selected_name": selected_name, 
            "ticker": ticker, "step_size": step_size,
            "models": models, "vol_models": vol_models,
            "buy_threshold": buy_threshold, "vol_limit": vol_limit,
            "show_bb": False, "show_rsi": False, "ma_settings": [], "show_signals": False
        }

def render_sidebar_actions(df, target_date_ts, config):
    with st.sidebar:
        st.write("---")
        st.subheader("🤖 AI 하이브리드 엔진 (Dynamic v3)")
        
        if st.session_state.hybrid_model.is_trained:
            st.success("✅ 균형 샘플링 모델 학습 완료")
        else:
            st.info("💡 모델을 학습시켜주세요.")

        def to_pct(val, base):
            return ((val - base) / base) * 100 if base != 0 else 0

        # --- 버튼 1: 모델 학습 (전략적 균형 샘플링 도입) ---
        if st.button("🚀 모델 전천후 학습 시작"):
            window_size, h = 60, config['step_size']
            max_idx = df.index.get_loc(target_date_ts)
            
            # [수정] 전체 과거 데이터(max_idx 이전)를 탐색 범위로 설정
            if max_idx < 150:
                st.error("학습을 위한 데이터가 충분하지 않습니다 (최소 150일 필요).")
            else:
                with st.spinner("과거 데이터에서 시장 국면별 샘플 추출 중..."):
                    temp_df = df.iloc[window_size : max_idx].copy()
                    
                    # 1. 등락률 기준으로 인덱스 분류 (Market Regime)
                    bear_idx = temp_df[temp_df['등락률'] < -2.0].index.tolist() # 급락장
                    bull_idx = temp_df[temp_df['등락률'] > 2.0].index.tolist()  # 급등장
                    side_idx = temp_df[(temp_df['등락률'] >= -2.0) & (temp_df['등락률'] <= 2.0)].index.tolist() # 횡보장
                    
                    # 2. 전략적 샘플링 비율 (방어형: 하락 데이터 40% 강제 할당)
                    sample_indices = []
                    target_total = 200 # 총 200개의 다양한 케이스 학습
                    
                    # 각 국면에서 추출할 개수 설정
                    configs = [
                        (bear_idx, int(target_total * 0.4)), # 하락장 80개
                        (side_idx, int(target_total * 0.3)), # 횡보장 60개
                        (bull_idx, int(target_total * 0.3))  # 상승장 60개
                    ]
                    
                    for pool, count in configs:
                        if len(pool) > 0:
                            # 해당 풀에서 가용한 만큼 랜덤 추출
                            chosen = random.sample(pool, min(len(pool), count))
                            sample_indices.extend([df.index.get_loc(c) for c in chosen])
                    
                    # 샘플들을 랜덤하게 섞음 (시계열 편향 제거)
                    random.shuffle(sample_indices)

                with st.spinner(f"{len(sample_indices)}개 균형 샘플로 학습 진행 중..."):
                    X_seq_list, X_static_list = [], []
                    y_actual_list, y_base_list = [], [] 

                    for idx in sample_indices:
                        if idx + h >= len(df) or idx < window_size: continue
                        
                        curr_p = df.iloc[idx - 1]['종가']
                        num_res = get_numerical_analysis(df.iloc[:idx]['종가'].values, h=h)
                        vol_res = get_volatility_models(df.iloc[:idx]['종가'].values)

                        euler_p = to_pct(num_res.get('euler', curr_p), curr_p)
                        rk4_p = to_pct(num_res.get('rk4', curr_p), curr_p)
                        newton_p = to_pct(num_res.get('newton', curr_p), curr_p)
                        base_p = (euler_p + rk4_p + newton_p) / 3

                        actual_p = to_pct(df.iloc[idx + h - 1]['종가'], curr_p)

                        static_feats = [
                            base_p, euler_p, rk4_p, newton_p,
                            vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0),
                            df.iloc[idx - 1].get('RSI', 50),
                            df.iloc[idx - 1].get('거래량_변동률', 0)
                        ]
                        
                        X_seq_list.append(df.iloc[idx - window_size : idx]['등락률'].values)
                        X_static_list.append(static_feats)
                        y_actual_list.append(actual_p)
                        y_base_list.append(base_p)
                    
                    st.session_state.hybrid_model.train(
                        np.nan_to_num(np.array(X_seq_list, dtype=np.float32).reshape(-1, window_size, 1)),
                        np.nan_to_num(np.array(X_static_list, dtype=np.float32)),
                        np.array(y_actual_list), 
                        np.array(y_base_list)
                    )
                    st.success("균형 학습 완료!"); st.rerun()

        # --- 버튼 2: 검증 (변동성 필터 및 문턱값 적용) ---
        if st.button("🔍 현재 모델로 이 구간 검증 (30일)"):
            if not st.session_state.hybrid_model.is_trained:
                st.error("모델 학습 필요")
            else:
                with st.spinner("다이내믹 검증 중..."):
                    st.session_state.history = []
                    h, val_idx = config['step_size'], df.index.get_loc(target_date_ts)
                    for i in range(val_idx - 30, val_idx + 1):
                        if i < 60 or i + h >= len(df): continue
                        curr_p = df.iloc[i - 1]['종가']
                        num_res = get_numerical_analysis(df.iloc[:i]['종가'].values, h=h)
                        vol_res = get_volatility_models(df.iloc[:i]['종가'].values)

                        euler_p = to_pct(num_res.get('euler', curr_p), curr_p)
                        rk4_p = to_pct(num_res.get('rk4', curr_p), curr_p)
                        newton_p = to_pct(num_res.get('newton', curr_p), curr_p)
                        base_p = (euler_p + rk4_p + newton_p) / 3

                        X_static_test = np.array([[
                            base_p, euler_p, rk4_p, newton_p, 
                            vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0), 
                            df.iloc[i - 1].get('RSI', 50), df.iloc[i - 1].get('거래량_변동률', 0)
                        ]], dtype=np.float32)
                        
                        pred_res = st.session_state.hybrid_model.predict(
                            np.nan_to_num(df.iloc[i - 60 : i]['등락률'].values.reshape(1, 60, 1)), 
                            np.nan_to_num(X_static_test)
                        )
                        
                        if pred_res is not None:
                            ai_residual, ai_weights = pred_res
                            w_euler, w_rk4, w_newton = ai_weights
                            
                            dynamic_base = (euler_p * w_euler) + (rk4_p * w_rk4) + (newton_p * w_newton)
                            final_pred_pct = dynamic_base + ai_residual
                            
                            actual_p = to_pct(df.iloc[i + h - 1]['종가'], curr_p)
                            current_vol = vol_res.get('egarch', 0)
    
                            # 🛡️ 영차의 방어적 매매 조건
                            is_confident = final_pred_pct > config['buy_threshold']
                            is_stable_market = current_vol < config['vol_limit']
                            is_buy = is_confident and is_stable_market
                            if is_buy:
                                strategy_return = actual_p  # 진입
                            else:
                                strategy_return = 0.0       # 관망 (MDD 방어)

                            st.session_state.history.append({
                                "date": df.index[i].date(), "actual": df.iloc[i + h - 1]['종가'], 
                                "pred": curr_p * (1 + final_pred_pct / 100),
                                "hit": (final_pred_pct > 0 and actual_p > 0) or (final_pred_pct < 0 and actual_p < 0),
                                "return": strategy_return,
                                "is_buy": is_buy
                            })
                    st.success("검증 완료"); st.rerun()
