import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 하이브리드 모델 입력 데이터 변환 함수
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    # 1. 시계열 데이터 변환 (LSTM용)
    if target_idx < window_size:
        return None, None
        
    seq_data = df.iloc[target_idx - window_size : target_idx]['종가'].values
    X_seq = seq_data.reshape(1, window_size, 1) # (샘플수, 타임스텝, 피처수)
    
    # 2. 정적 수치 데이터 변환 (XGBoost용)
    X_static = np.array([[
        predictions.get('euler', 0),
        predictions.get('rk4', 0),
        predictions.get('newton', 0),
        vol_results.get('egarch', 0),
        vol_results.get('gjr_garch', 0)
    ]])
    
    return X_seq, X_static

# 인자에 vol_results와 predictions를 추가하여 main.py로부터 전달받습니다.
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 {config['target_date']} 분석 결과")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    # 데이터 인덱싱 및 가격 추출
    target_idx = df.index.get_loc(target_date_ts)
    actual_price = df.loc[target_date_ts, '종가']
    prev_price = df.iloc[target_idx - config['step_size']]['종가']

    # --- 1. 변동성 모델 섹션 (사이드바 연동) ---
    if vol_results and any(config['vol_models'].values()):
        st.markdown("### 📉 변동성 분석 (Volatility Models)")
        v_col1, v_col2 = st.columns(2)
        
        if config['vol_models'].get('egarch'):
            with v_col1:
                st.metric("EGARCH 변동성", f"{vol_results['egarch']:.2f}%")
                st.caption("하락장에 민감한 변동성 지수")
        
        if config['vol_models'].get('gjr_garch'):
            with v_col2:
                st.metric("GJR-GARCH 변동성", f"{vol_results['gjr_garch']:.2f}%")
                st.caption("상승/하락 비대칭 리스크")
        st.divider()

    # --- [NEW] 하이브리드 모델 예측 실행 ---
    hybrid_pred_val = None
    if config['models'].get('hybrid') and vol_results and predictions:
        X_seq, X_static = prepare_hybrid_input(df, target_idx, vol_results, predictions)
        
        if X_seq is not None and 'hybrid_model' in st.session_state:
            hybrid_pred_val = st.session_state.hybrid_model.predict(X_seq, X_static)[0]
        elif X_seq is None:
            st.warning("⚠️ 하이브리드 모델을 실행하기 위한 과거 데이터(최소 60일)가 부족합니다.")

    # --- 2. 예측 모델 비교 섹션 ---
    if predictions:
        st.markdown("### 🧮 예측 모델 비교")
        comparison_data = []

        if hybrid_pred_val is not None:
            error_rate = ((actual_price - hybrid_pred_val) / actual_price) * 100
            is_hit = (hybrid_pred_val > prev_price and actual_price > prev_price) or \
                     (hybrid_pred_val < prev_price and actual_price < prev_price)
            comparison_data.append({
                "모델": "HYBRID (LSTM+XGB)", 
                "예측가": f"{int(hybrid_pred_val):,}원",
                "실제가": f"{int(actual_price):,}원",
                "오차율": f"{error_rate:+.2f}%", 
                "방향": "🟢 적중" if is_hit else "🔴 실패"
            })
            
            # --- [성능 기록 저장] ---
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            # 중복 날짜 저장 방지
            if not any(h['date'] == config['target_date'] for h in st.session_state.history):
                st.session_state.history.append({
                    "date": config['target_date'],
                    "actual": actual_price,
                    "pred": hybrid_pred_val,
                    "hit": is_hit
                })

        for m, active in config['models'].items():
            if not active or m == 'hybrid': continue
            
            pred_val = predictions.get(m)
            if pred_val is None: continue

            if m == "simpson":
                comparison_data.append({
                    "모델": "SIMPSON", 
                    "예측/에너지": f"{pred_val:.4f}", 
                    "실제가": f"{int(actual_price):,}원", 
                    "오차율": "N/A", 
                    "방향": "-"
                })
            else:
                error_rate = ((actual_price - pred_val) / actual_price) * 100
                is_hit_m = (pred_val > prev_price and actual_price > prev_price) or \
                           (pred_val < prev_price and actual_price < prev_price)
                
                comparison_data.append({
                    "모델": m.upper(), 
                    "예측가": f"{int(pred_val):,}원", 
                    "실제가": f"{int(actual_price):,}원",
                    "오차율": f"{error_rate:+.2f}%", 
                    "방향": "🟢 적중" if is_hit_m else "🔴 실패"
                })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            styled_df = comp_df.style.map(
                lambda v: 'color: #00C805; font-weight: bold' if '🟢' in str(v) else 
                          ('color: #FF4B4B; font-weight: bold' if '🔴' in str(v) else ''), 
                subset=['방향']
            )
            st.table(styled_df)
            st.write(f"**전일 대비 변화:** {int(prev_price):,}원 → {int(actual_price):,}원")
        else:
            st.info("💡 왼쪽 사이드바에서 예측 모델을 선택해 주세요.")

    # --- 3. [NEW] AI 모델 성능 검증 섹션 ---
    if 'history' in st.session_state and len(st.session_state.history) > 0:
        st.divider()
        st.markdown("### 🏆 AI 하이브리드 모델 성능 검증")
        
        hist_df = pd.DataFrame(st.session_state.history).sort_values("date")
        
        col_gauge, col_trend = st.columns([1, 2])
        
        # (1) 방향 적중률 게이지
        with col_gauge:
            hit_rate = (hist_df['hit'].sum() / len(hist_df)) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = hit_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "방향 적중률 (%)", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00C805"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FF4B4B"},
                        {'range': [50, 100], 'color': "#E8F5E9"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # (2) 예측 vs 실제 추세 차트
        with col_trend:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=hist_df['date'], y=hist_df['actual'],
                mode='lines+markers', name="실제 종가",
                line=dict(color="#1C83E1", width=3)
            ))
            fig_trend.add_trace(go.Scatter(
                x=hist_df['date'], y=hist_df['pred'],
                mode='lines+markers', name="AI 예측가",
                line=dict(color="#FF4B4B", dash='dot', width=2)
            ))
            fig_trend.update_layout(
                title="AI 예측 vs 실제 종가 추세",
                xaxis_title="날짜",
                yaxis_title="가격(원)",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
        st.caption(f"💡 현재까지 총 {len(hist_df)}회의 분석 데이터가 누적되었습니다. 날짜를 변경하며 백테스팅을 진행하면 지표가 업데이트됩니다.")
