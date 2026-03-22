import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
                    
                    st.markdown("### 📋 종목 분석 리포트")
                    
                    up_prob = min(max(50 + (final_pred_pct * 15), 10), 95)
                    prob_color = "#00C805" if up_prob >= 50 else "#FF4B4B"
                    
                    st.markdown(f"**AI 상승 확률:** <span style='color:{prob_color}; font-size:1.3em; font-weight:bold;'>{up_prob:.0f}%</span>", unsafe_allow_html=True)
                    st.write("") 
                    
                    vol_change = df.iloc[target_idx].get('거래량_변동률', 0)
                    rsi_val = df.iloc[target_idx].get('RSI', 50)
                    egarch_val = vol_results.get('egarch', 0) if vol_results else 0
                    
                    reasons = []
                    
                    if vol_change > 50:
                        reasons.append(f"**거래량 급증 ({vol_change:+.0f}%):** 전일 대비 거래량이 크게 늘어나며 새로운 매수 자금이 유입되는 패턴이 포착되었습니다.")
                    elif vol_change < -30:
                        reasons.append(f"**거래량 축소 ({vol_change:+.0f}%):** 매도 물량이 마르면서 방향성을 결정하기 직전의 응축 단계입니다.")
                    else:
                        reasons.append(f"**거래량 안정 유지:** 평소 수준의 거래량을 유지하며 기존의 가격 추세를 탄탄하게 지지하고 있습니다.")
                        
                    if egarch_val < 5.0 and rsi_val < 40:
                        reasons.append(f"**반등 에너지 축적 (RSI {rsi_val:.1f}):** 주가가 바닥을 다지고 다시 상승할 수 있는 에너지가 충분히 모인 상태입니다.")
                    elif rsi_val > 65:
                        reasons.append(f"**강한 상승 모멘텀 (RSI {rsi_val:.1f}):** 현재 시장의 매수 심리가 매우 강해 추세가 위로 열려 있습니다.")
                    else:
                        reasons.append(f"**안정적 수급 (RSI {rsi_val:.1f}):** 시장이 과열되거나 침체되지 않고 정상적인 수급 흐름을 보이고 있습니다.")
                        
                    if w_rk4 > 0.4:
                        reasons.append(f"**단기 가속도 진입 (RK4 비중 {w_rk4*100:.0f}%):** 알고리즘 분석 결과, 주가가 일시적으로 강하게 튀어 오를 수 있는 가속도 구간에 진입했습니다.")
                    elif w_newton > 0.4:
                        reasons.append(f"**추세 전환점 도달 (Newton 비중 {w_newton*100:.0f}%):** 기존의 하락(또는 상승) 흐름이 반대로 뒤집히는 결정적인 변곡점이 포착되었습니다.")
                    else:
                        reasons.append(f"**추세 유지 (Euler 비중 {w_euler*100:.0f}%):** 갑작스러운 변동보다는, 기존의 안정적인 우상향(우하향) 궤도를 그대로 따라가고 있습니다.")

                    st.markdown("**💡 AI 주가 예측 결과의 근거**")
                    for i, reason in enumerate(reasons, 1):
                        st.markdown(f"{i}. {reason}")
                        
                    st.write("")

                    risks = []
                    if egarch_val > 10.0:
                        risks.append(f"**돌발 하락 주의 (EGARCH {egarch_val:.1f}):** 변동성이 커져 있어, 작은 악재에도 주가가 크게 흔들릴 수 있는 불안정한 상태입니다.")
                    if rsi_val > 70:
                        risks.append(f"**단기 고점 과열 (RSI {rsi_val:.1f}):** 매수세가 너무 몰려 있어 차익 실현(매도) 물량이 쏟아지며 단기 조정을 받을 수 있습니다.")
                    if final_pred_pct < 0:
                        risks.append("**하방 압력 우세:** 알고리즘이 예측한 최종 물리적 궤도가 꺾여 있어 추가 하락 가능성을 열어두어야 합니다.")
                        
                    if not risks:
                        risks.append("현재 뚜렷한 기술적 악재나 급락 리스크 요인이 보이지 않습니다.")

                    st.markdown("**⚠️ 리스크 및 주의사항**")
                    for risk in risks:
                        st.markdown(f"- {risk}")

                    if 'history' in st.session_state and len(st.session_state.history) > 0:
                        st.divider()
                        render_performance_visuals()
                else:
                    st.error("⚠️ 예측 데이터를 생성하는 중 오류가 발생했습니다.")
            else:
                st.info("💡 과거 데이터 부족으로 AI 분석이 불가능합니다.")

def render_performance_visuals():
    hist_df = pd.DataFrame(st.session_state.history).sort_values("date")
    
    returns = hist_df['return'] / 100
    cum_returns = (1 + returns).cumprod()
    hist_df['cum_return_pct'] = (cum_returns - 1) * 100
    
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    mdd = drawdown.min() * 100
    
    hit_rate = (hist_df['hit'].sum() / len(hist_df)) * 100
    final_profit = hist_df['cum_return_pct'].iloc[-1]

    st.markdown(f"### 📊 {len(hist_df)}일 수익성 검증 리포트")
    m1, m2, m3 = st.columns(3)
    m1.metric("방향 적중률", f"{hit_rate:.1f}%")
    m2.metric("누적 수익률", f"{final_profit:+.2f}%", delta=f"{hist_df['return'].iloc[-1]:+.2f}% (최근)")
    m3.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%", help="검증 기간 중 최고점 대비 가장 많이 하락했던 비율입니다.")

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
