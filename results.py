import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 하이브리드 모델 입력 데이터 변환 (기존과 동일)
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    if '등락률' not in df.columns:
        df['등락률'] = df['종가'].pct_change() * 100
    if '거래량_변동률' not in df.columns:
        df['거래량_변동률'] = df['거래량'].pct_change() * 100
    
    if target_idx < window_size:
        return None, None, 0, 0, 0
        
    try:
        seq_data = df.iloc[target_idx - window_size + 1 : target_idx + 1]['등락률'].values
        X_seq = np.array(seq_data, dtype=np.float32).reshape(1, window_size, 1)
        
        curr_p = df.iloc[target_idx]['종가']
        def to_pct(val, base):
            return ((val - base) / base) * 100 if base != 0 else 0

        euler_p = to_pct(predictions.get('euler', curr_p), curr_p)
        rk4_p = to_pct(predictions.get('rk4', curr_p), curr_p)
        newton_p = to_pct(predictions.get('newton', curr_p), curr_p)
        base_p = (euler_p + rk4_p + newton_p) / 3

        X_static = np.array([[
            base_p, euler_p, rk4_p, newton_p,
            float(vol_results.get('egarch', 0)),
            float(vol_results.get('gjr_garch', 0)),
            float(df.iloc[target_idx].get('RSI', 50)),
            float(df.iloc[target_idx].get('거래량_변동률', 0))
        ]], dtype=np.float32)
        
        return np.nan_to_num(X_seq), np.nan_to_num(X_static), euler_p, rk4_p, newton_p
    except Exception as e:
        return None, None, 0, 0, 0

# 2. 메인 결과 렌더링 함수 (기존과 동일)
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 AI 분석 리포트 (Dynamic v3)")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    target_idx = df.index.get_loc(target_date_ts)
    prev_price = df.iloc[target_idx]['종가']

    if 'hybrid_model' in st.session_state:
        if not st.session_state.hybrid_model.is_trained:
            st.info("🚀 분석 준비 완료! 왼쪽 하단의 '모델 전천후 학습 시작'을 클릭하시면 AI 리포트가 생성됩니다.")
            return
            
        if config['models'].get('hybrid'):
            X_seq, X_static, euler_p, rk4_p, newton_p = prepare_hybrid_input(df, target_idx, vol_results, predictions)
            
            if X_seq is not None:
                pred_res = st.session_state.hybrid_model.predict(X_seq, X_static)
                
                if pred_res is not None and len(pred_res) == 2:
                    ai_residual, ai_weights = pred_res
                    w_euler, w_rk4, w_newton = ai_weights
                    
                    dynamic_base = (euler_p * w_euler) + (rk4_p * w_rk4) + (newton_p * w_newton)
                    final_pred_pct = dynamic_base + ai_residual
                    hybrid_pred_val = prev_price * (1 + final_pred_pct / 100)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("최종 예측 등락", f"{final_pred_pct:+.2f}%", delta=f"AI보정: {ai_residual:+.2f}%")
                    with c2:
                        st.metric("예측 목표가", f"{int(hybrid_pred_val):,}원")
                    with c3:
                        if target_idx + 1 < len(df):
                            actual_next = df.iloc[target_idx + 1]['종가']
                            error_rate = ((actual_next - hybrid_pred_val) / actual_next) * 100
                            st.metric("실제 오차율", f"{error_rate:+.2f}%")
                        else:
                            st.metric("예측 오차율", "데이터 대기 중")
                    
                    st.write("---")
                    st.markdown("🤖 **AI의 실시간 모델 신뢰도 분석 (Dynamic Gating)**")
                    w_col1, w_col2, w_col3 = st.columns(3)
                    w_col1.metric("Euler (안정/추세)", f"{w_euler*100:.1f}%")
                    w_col2.metric("RK4 (정밀 추세)", f"{w_rk4*100:.1f}%")
                    w_col3.metric("Newton (변곡점)", f"{w_newton*100:.1f}%")
                    
                    best_model = max([("Euler", w_euler), ("RK4", w_rk4), ("Newton", w_newton)], key=lambda x: x[1])
                    st.caption(f"💡 **AI 판단**: 과거 패턴 분석 결과, 현재 시장 흐름은 **{best_model[0]}** 모델의 계산과 가장 유사하여 해당 비중을 높였습니다.")

                    if 'history' in st.session_state and len(st.session_state.history) > 0:
                        st.divider()
                        render_performance_visuals()
                else:
                    st.error("⚠️ 예측 데이터를 생성하는 중 오류가 발생했습니다.")
            else:
                st.info("💡 과거 데이터 부족으로 AI 분석이 불가능합니다.")

# [업그레이드] 수익성 검증 시각화 함수
def render_performance_visuals():
    hist_df = pd.DataFrame(st.session_state.history).sort_values("date")
    
    # 1. 수익성 지표 계산
    # 누적 수익률 (복리: 1.1 * 0.9 = 0.99 방식)
    returns = hist_df['return'] / 100
    cum_returns = (1 + returns).cumprod()
    hist_df['cum_return_pct'] = (cum_returns - 1) * 100
    
    # MDD 계산 (최고점 대비 최대 하락폭)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    mdd = drawdown.min() * 100
    
    hit_rate = (hist_df['hit'].sum() / len(hist_df)) * 100
    final_profit = hist_df['cum_return_pct'].iloc[-1]

    # 2. 요약 지표 출력
    st.markdown(f"### 📊 {len(hist_df)}일 수익성 검증 리포트")
    m1, m2, m3 = st.columns(3)
    m1.metric("방향 적중률", f"{hit_rate:.1f}%")
    m2.metric("누적 수익률", f"{final_profit:+.2f}%", delta=f"{hist_df['return'].iloc[-1]:+.2f}% (최근)")
    m3.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%", help="검증 기간 중 최고점 대비 가장 많이 하락했던 비율입니다.")

    # 3. 누적 수익률 차트 (추가)
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=hist_df['date'], 
        y=hist_df['cum_return_pct'], 
        fill='tozeroy', 
        name="누적 수익률",
        line=dict(color="#00C805" if final_profit >= 0 else "#FF4B4B", width=3)
    ))
    fig_cum.update_layout(
        title="AI 전략 누적 수익률 추이 (%)",
        height=250, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        yaxis=dict(ticksuffix="%")
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # 4. 기존 실제가 vs 예측가 차트 (하단으로 이동)
    col_g, col_t = st.columns([1, 2])
    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = hit_rate,
            title = {'text': "방향 적중률", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]}, 
                'bar': {'color': "#1C83E1"},
                'threshold': {'line': {'color': "white", 'width': 3}, 'value': 60}
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_t:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['actual'], name="실제가", line=dict(color="#1C83E1")))
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['pred'], name="AI예측", line=dict(color="#FF4B4B", dash='dot')))
        fig_trend.update_layout(title="예측 트렌드 점검", height=200, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig_trend, use_container_width=True)
