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
        
        val_days = st.slider(
            "📅 검증 기간 설정 (일)", 10, 365, 30, 
            help="백테스팅을 진행할 과거 기간을 설정합니다. (최대 1년)"
        )
        
        st.write("---")
        market_type = st.radio("🌐 시장 선택", ["주식 (한국)", "가상화폐 (KRW)"], horizontal=True)
        
        buy_threshold = st.slider(
            "🎯 매수 문턱값 (%)", 1.0, 3.0, 1.0, step=0.1,
            help="AI의 상승 예측치가 이 값보다 높을 때만 매수합니다."
        )
        
        vol_limit = st.slider(
            "⚠️ 변동성 제한 (GARCH)", 3.0, 15.0, 10.0, step=0.5,
            help="시장 변동성(EGARCH)이 이 수치보다 높으면 '위험'으로 간주하고 매수하지 않습니다."
        )

        if market_type == "주식 (한국)":
            search_word = st.text_input("🔍 종목명 또는 코드 입력", "삼성전자").strip()
            ticker_dict = get_tickers()
            default_name, default_ticker = "삼성전자", "005930"
            
            # 🚀 [디버깅] 정상적으로 로드된 종목 개수 표시
            if len(ticker_dict) > 100:
                st.caption(f"✅ 총 {len(ticker_dict):,}개 종목 로드 완료")
            else:
                st.error("⚠️ 데이터 서버 접근 실패 (KRX 일시 차단 의심)")
                
        else:
            search_word = st.text_input("🔍 코인 이름 또는 심볼 입력", "BTC").strip()
            ticker_dict = get_coin_tickers()
            default_name, default_ticker = "BTC", "KRW-BTC"
            
        clean_query = search_word.lower()
        matched_names = [
            name for name, tk in ticker_dict.items() 
            if clean_query in name.lower() or clean_query in tk.lower()
        ]
        
        if matched_names:
            selected_name = st.selectbox("🎯 검색 결과 중 선택", sorted(matched_names))
            ticker = ticker_dict[selected_name]
        elif search_word:
            st.error(f"❌ '{search_word}'와(과) 일치하는 결과가 없습니다.")
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
            "val_days": val_days,
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

        if st.button("🚀 모델 전천후 학습 시작"):
            window_size, h = 60, config['step_size']
            max_idx = df.index.get_loc(target_date_ts)
            
            if max_idx < 150:
                st.error("학습을 위한 데이터가 충분하지 않습니다 (최소 150일 필요).")
            else:
                with st.spinner("과거 데이터에서 시장 국면별 샘플 추출 중..."):
                    temp_df = df.iloc[window_size : max_idx].copy()
                    
                    bear_idx = temp_df[temp_df['등락률'] < -2.0].index.tolist()
                    bull_idx = temp_df[temp_df['등락률'] > 2.0].index.tolist() 
                    side_idx = temp_df[(temp_df['등락률'] >= -2.0) & (temp_df['등락률'] <= 2.0)].index.tolist()
                    
                    sample_indices = []
                    target_total = 200 
                    
                    configs = [
                        (bear_idx, int(target_total * 0.4)), 
                        (side_idx, int(target_total * 0.3)), 
                        (bull_idx, int(target_total * 0.3))  
                    ]
                    
                    for pool, count in configs:
                        if len(pool) > 0:
                            chosen = random.sample(pool, min(len(pool), count))
                            sample_indices.extend([df.index.get_loc(c) for c in chosen])
                    
                    random.shuffle(sample_indices)

                with st.spinner(f"{len(sample_indices)}개 균형 샘플로 학습 진행 중..."):
                    X_seq_list, X_static_list = [], []
                    y_actual_list, y_base_list = [], [] 

                    for idx in sample_indices:
                        if idx + h >= len(df) or idx < window_size: continue
                        
                        curr_p = df.iloc[idx]['종가']
                        num_res = get_numerical_analysis(df.iloc[:idx+1]['종가'].values, h=h)
                        vol_res = get_volatility_models(df.iloc[:idx+1]['종가'].values)

                        euler_p = to_pct(num_res.get('euler', curr_p), curr_p)
                        rk4_p = to_pct(num_res.get('rk4', curr_p), curr_p)
                        newton_p = to_pct(num_res.get('newton', curr_p), curr_p)
                        base_p = (euler_p + rk4_p + newton_p) / 3

                        actual_p = to_pct(df.iloc[idx + h]['종가'], curr_p)

                        static_feats = [
                            base_p, euler_p, rk4_p, newton_p,
                            vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0),
                            df.iloc[idx].get('RSI', 50),
                            df.iloc[idx].get('거래량_변동률', 0)
                        ]
                        
                        X_seq_list.append(df.iloc[idx - window_size + 1 : idx + 1]['등락률'].values)
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

        val_label = f"🔍 현재 모델로 이 구간 검증 ({config['val_days']}일)" 
        if st.button(val_label):
            if not st.session_state.hybrid_model.is_trained:
                st.error("모델 학습 필요")
            else:
                with st.spinner(f"{config['val_days']}일간 다이내믹 검증 중..."):
                    st.session_state.history = []
                    h, val_idx = config['step_size'], df.index.get_loc(target_date_ts)
                    start_idx = val_idx - config['val_days']
                    
                    for i in range(start_idx, val_idx + 1):
                        if i < 60 or i + h >= len(df): continue
                        curr_p = df.iloc[i]['종가']
                        num_res = get_numerical_analysis(df.iloc[:i+1]['종가'].values, h=h)
                        vol_res = get_volatility_models(df.iloc[:i+1]['종가'].values)

                        euler_p = to_pct(num_res.get('euler', curr_p), curr_p)
                        rk4_p = to_pct(num_res.get('rk4', curr_p), curr_p)
                        newton_p = to_pct(num_res.get('newton', curr_p), curr_p)
                        base_p = (euler_p + rk4_p + newton_p) / 3

                        X_static_test = np.array([[
                            base_p, euler_p, rk4_p, newton_p, 
                            vol_res.get('egarch', 0), vol_res.get('gjr_garch', 0), 
                            df.iloc[i].get('RSI', 50),
                            df.iloc[i].get('거래량_변동률', 0)
                        ]], dtype=np.float32)
                        
                        pred_res = st.session_state.hybrid_model.predict(
                            np.nan_to_num(df.iloc[i - 59 : i + 1]['등락률'].values.reshape(1, 60, 1)), 
                            np.nan_to_num(X_static_test)
                        )
                        
                        if pred_res is not None:
                            ai_residual, ai_weights = pred_res
                            w_euler, w_rk4, w_newton = ai_weights
                            
                            dynamic_base = (euler_p * w_euler) + (rk4_p * w_rk4) + (newton_p * w_newton)
                            final_pred_pct = dynamic_base + ai_residual

                            future_price = df.iloc[i+h]['종가']
                            actual_p = to_pct(future_price, curr_p)
                            current_vol = vol_res.get('egarch', 0)
    
                            is_confident = final_pred_pct > config['buy_threshold']
                            is_stable_market = current_vol < config['vol_limit']
                            is_buy = is_confident and is_stable_market
                            
                            strategy_return = actual_p if is_buy else 0.0

                            st.session_state.history.append({
                                "date": df.index[i+h].date(), "actual": future_price, 
                                "pred": curr_p * (1 + final_pred_pct / 100),
                                "hit": (final_pred_pct > 0 and actual_p > 0) or (final_pred_pct < 0 and actual_p < 0),
                                "return": strategy_return,
                                "is_buy": is_buy
                            })
                    st.success("검증 완료"); st.rerun()
