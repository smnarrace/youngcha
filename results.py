import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 하이브리드 모델 입력 데이터 변환 (등락률 기반)
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    # [변경] 종가 대신 '등락률' 컬럼 사용
    if '등락률' not in df.columns:
        df['등락률'] = df['종가'].pct_change() * 100
        df = df.fillna(0)

    if target_idx < window_size:
        return None, None
        
    # LSTM용: 등락률 시퀀스 추출
    seq_data = df.iloc[target_idx - window_size + 1 : target_idx + 1]['등락률'].values
    X_seq = np.array(seq_data, dtype=np.float32).reshape(1, window_size, 1)
    
    # XGBoost용: 수치해석 및 변동성 데이터
    X_static = np.array([[
        predictions.get('euler', 0),
        predictions.get('rk4', 0),
        predictions.get('newton', 0),
        vol_results.get('egarch', 0),
        vol_results.get('gjr_garch', 0)
    ]], dtype=np.float32)
    
    return np.nan_to_num(X_seq), np.nan_to_num(X_static)

# 2. 메인 결과 렌더링 함수
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 {config['target_date']} AI 리포트")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 해당 날짜의 데이터가 없습니다.")
        return

    target_idx = df.index.get_loc(target_date_ts)
    actual_price = df.loc[target_date_ts, '종가']
    # [변경] 등락률 복구를 위해 전일 종가 확보
    prev_price = df.iloc[target_idx - 1]['종가']

    # --- 1. AI 핵심 지표 (Metric) ---
    hybrid_pred_val = None
    if config['models'].get('hybrid') and 'hybrid_model' in st.session_state:
        X_seq, X_static = prepare_hybrid_input(df, target_idx, vol_results, predictions)
        
        if X_seq is not None:
            # 모델 예측 결과는 % (예: 1.25)
            pred_pct = st.session_state.hybrid_model.predict(X_seq, X_static)[0]
            # [핵심] %를 가격으로 복구: $P_{pred} = P_{prev} \times (1 + \frac{r_{pred}}{100})$
            hybrid_pred_val = prev_price * (1 + pred_pct / 100)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("AI 예측 등락", f"{pred_pct:+.2f}%")
            with c2:
                st.metric("AI 목표가", f"{int(hybrid_pred_val):,}원")
            with c3:
                error_rate = ((actual_price - hybrid_pred_val) / actual_price) * 100
                st.metric("예측 오차율", f"{error_rate:+.2f}%")
        else:
            st.info("💡 과거 데이터가 부족하여 하이브리드 분석을 건너뜁니다.")

    # --- 2. 모델 성능 검증 (게이지 & 추세) ---
    if 'history' in st.session_state and len(st.session_state.history) > 0:
        st.divider()
        render_performance_section(actual_price, hybrid_pred_val, config)

# 3. 게이지 및 추세 차트 분리
def render_performance_section(actual_price, hybrid_pred_val, config):
    st.markdown("### 🏆 AI 엔진 신뢰도 검증")
    
    # 세션 기록 업데이트 (중복 방지)
    if hybrid_pred_val is not None:
        is_hit = (hybrid_pred_val > actual_price * 0.999 and actual_price > actual_price * 0.999) # 예시 로직 수정 필요
        # 실제 방향 적중 판정
        # 실제가가 전일보다 올랐는데 예측가도 전일보다 높으면 적중
        # history 저장 시 main.py에서 이미 hit 계산을 하므로 여기선 출력에 집중
    
    hist_df = pd.DataFrame(st.session_state.history).sort_values("date")
    
    col_gauge, col_trend = st.columns([1, 2])
    
    with col_gauge:
        hit_rate = (hist_df['hit'].sum() / len(hist_df)) * 100
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = hit_rate,
            title = {'text': "방향 적중률 (%)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00C805" if hit_rate >= 60 else "#FF4B4B"},
                'steps': [{'range': [0, 50], 'color': "#f4f4f4"}],
                'threshold': {'line': {'color': "black", 'width': 3}, 'value': 70}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_trend:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['actual'], name="실제가", line=dict(color="#1C83E1")))
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['pred'], name="AI예측", line=dict(color="#FF4B4B", dash='dot')))
        fig_trend.update_layout(title="최근 예측 트렌드", height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)
