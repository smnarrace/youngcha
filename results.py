import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 하이브리드 모델 입력 데이터 변환 (8개 피처 & 잔차 학습용)
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    if '등락률' not in df.columns:
        df['등락률'] = df['종가'].pct_change() * 100
    if '거래량_변동률' not in df.columns:
        df['거래량_변동률'] = df['거래량'].pct_change() * 100
    
    if target_idx < window_size:
        return None, None
        
    try:
        seq_data = df.iloc[target_idx - window_size + 1 : target_idx + 1]['등락률'].values
        X_seq = np.array(seq_data, dtype=np.float32).reshape(1, window_size, 1)
        
        curr_p = df.iloc[target_idx]['종가']
        def to_pct(val, base):
            return ((val - base) / base) * 100 if base != 0 else 0

        # [수정] 3대 모델 평균 베이스라인 계산
        euler_p = to_pct(predictions.get('euler', curr_p), curr_p)
        rk4_p = to_pct(predictions.get('rk4', curr_p), curr_p)
        newton_p = to_pct(predictions.get('newton', curr_p), curr_p)
        base_p = (euler_p + rk4_p + newton_p) / 3

        # [수정] 8개 피처 구성 (베이스라인 + 개별3 + 변동성2 + RSI + 거래량)
        X_static = np.array([[
            base_p, euler_p, rk4_p, newton_p,
            float(vol_results.get('egarch', 0)),
            float(vol_results.get('gjr_garch', 0)),
            float(df.iloc[target_idx].get('RSI', 50)),
            float(df.iloc[target_idx].get('거래량_변동률', 0))
        ]], dtype=np.float32)
        
        return np.nan_to_num(X_seq), np.nan_to_num(X_static), base_p
    except Exception as e:
        return None, None, 0

# 2. 메인 결과 렌더링 함수
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 AI 분석 리포트 (Residual v2)")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    target_idx = df.index.get_loc(target_date_ts)
    prev_price = df.iloc[target_idx]['종가']

    if 'hybrid_model' in st.session_state:
        if not st.session_state.hybrid_model.is_trained:
            st.info("🚀 분석 준비 완료! 왼쪽 하단의 '모델 전천후 학습 시작'을 클릭하시면 잔차 보정 리포트가 생성됩니다.")
            return
            
        if config['models'].get('hybrid'):
            # [수정] 베이스라인 값(base_p)도 함께 받아옴
            X_seq, X_static, base_p = prepare_hybrid_input(df, target_idx, vol_results, predictions)
            
            if X_seq is not None:
                # AI가 예측한 것은 '오차(Residual)' 값입니다.
                pred_res = st.session_state.hybrid_model.predict(X_seq, X_static)
                
                if pred_res is not None and len(pred_res) > 0:
                    ai_residual = pred_res[0] 
                    # [핵심] 최종 등락률 = 수학적 평균(base_p) + AI의 보정치(ai_residual)
                    final_pred_pct = base_p + ai_residual
                    hybrid_pred_val = prev_price * (1 + final_pred_pct / 100)
                    
                    # 결과 지표 출력
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        # 보정치를 help 툴팁으로 보여주면 더 전문적입니다.
                        st.metric("최종 예측 등락", f"{final_pred_pct:+.2f}%", 
                                  delta=f"AI보정: {ai_residual:+.2f}%")
                    with c2:
                        st.metric("예측 목표가", f"{int(hybrid_pred_val):,}원")
                    with c3:
                        if target_idx + 1 < len(df):
                            actual_next = df.iloc[target_idx + 1]['종가']
                            error_rate = ((actual_next - hybrid_pred_val) / actual_next) * 100
                            st.metric("실제 오차율", f"{error_rate:+.2f}%")
                        else:
                            st.metric("예측 오차율", "데이터 대기 중")
                    
                    # 상세 분석 가이드 (잔차 학습의 묘미)
                    with st.expander("🔍 하이브리드 분석 디테일"):
                        st.write(f"1. **수학적 베이스라인**: {base_p:+.2f}% (Euler, RK4, Newton 평균)")
                        st.write(f"2. **AI 오차 보정치**: {ai_residual:+.2f}% (Attention LSTM + XGBoost)")
                        st.write(f"**최종 결론**: 수학적 흐름에 AI의 패턴 인식을 더해 **{final_pred_pct:+.2f}%**의 변동을 예상합니다.")

                    if 'history' in st.session_state and len(st.session_state.history) > 0:
                        st.divider()
                        render_performance_visuals()
                else:
                    st.error("⚠️ 예측 데이터를 생성하는 중 오류가 발생했습니다.")
            else:
                st.info("💡 과거 데이터 부족으로 AI 분석이 불가능합니다.")

# (render_performance_visuals 함수는 기존과 동일하게 유지)
def render_performance_visuals():
    hist_df = pd.DataFrame(st.session_state.history).sort_values("date")
    hit_rate = (hist_df['hit'].sum() / len(hist_df)) * 100
    
    col_g, col_t = st.columns([1, 2])
    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = hit_rate,
            title = {'text': "방향 적중률", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]}, 
                'bar': {'color': "#00C805" if hit_rate >= 55 else "#FF4B4B"},
                'steps': [{'range': [0, 50], 'color': "#f4f4f4"}],
                'threshold': {'line': {'color': "black", 'width': 3}, 'value': 60}
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_t:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['actual'], name="실제가", line=dict(color="#1C83E1")))
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['pred'], name="AI예측", line=dict(color="#FF4B4B", dash='dot')))
        fig_trend.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig_trend, use_container_width=True)
