import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 하이브리드 모델 입력 데이터 변환 (7개 피처 & % 변환 적용)
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

        X_static = np.array([[
            to_pct(predictions.get('euler', curr_p), curr_p),
            to_pct(predictions.get('rk4', curr_p), curr_p),
            to_pct(predictions.get('newton', curr_p), curr_p),
            float(vol_results.get('egarch', 0)),
            float(vol_results.get('gjr_garch', 0)),
            float(df.iloc[target_idx].get('RSI', 50)),
            float(df.iloc[target_idx].get('거래량_변동률', 0))
        ]], dtype=np.float32)
        
        return np.nan_to_num(X_seq), np.nan_to_num(X_static)
    except Exception as e:
        # 이 에러는 진짜 시스템 에러일 때만 띄웁니다.
        return None, None

# 2. 메인 결과 렌더링 함수
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 AI 분석 리포트")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    target_idx = df.index.get_loc(target_date_ts)
    prev_price = df.iloc[target_idx]['종가']

    # --- [수정 포인트] 모델 학습 여부 먼저 체크 ---
    if 'hybrid_model' in st.session_state:
        # A. 아직 학습되지 않은 경우: 빨간 에러 대신 파란 안내창
        if not st.session_state.hybrid_model.is_trained:
            st.info("🚀 분석 준비 완료! 왼쪽 사이드바 하단에서 '모델 전천후 학습 시작' 버튼을 클릭하시면 AI 리포트가 생성됩니다.")
            return # 더 이상 아래 로직을 타지 않고 종료
            
        # B. 학습이 된 경우에만 예측 프로세스 시작
        if config['models'].get('hybrid'):
            X_seq, X_static = prepare_hybrid_input(df, target_idx, vol_results, predictions)
            
            if X_seq is not None:
                pred_res = st.session_state.hybrid_model.predict(X_seq, X_static)
                
                if pred_res is not None and len(pred_res) > 0:
                    pred_pct = pred_res[0] 
                    hybrid_pred_val = prev_price * (1 + pred_pct / 100)
                    
                    # 결과 지표 출력
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("AI 예측 등락", f"{pred_pct:+.2f}%")
                    with c2:
                        st.metric("AI 목표가", f"{int(hybrid_pred_val):,}원")
                    with c3:
                        if target_idx + 1 < len(df):
                            actual_next = df.iloc[target_idx + 1]['종가']
                            error_rate = ((actual_next - hybrid_pred_val) / actual_next) * 100
                            st.metric("실제 오차율", f"{error_rate:+.2f}%")
                        else:
                            st.metric("예측 오차율", "데이터 대기 중")
                    
                    # 성능 검증 섹션
                    if 'history' in st.session_state and len(st.session_state.history) > 0:
                        st.divider()
                        render_performance_visuals()
                else:
                    # 학습은 됐는데 예측값이 안 나온 경우 (드문 케이스)
                    st.error("⚠️ 예측 데이터를 생성하는 중 오류가 발생했습니다. 재학습을 권장합니다.")
            else:
                st.info("💡 과거 데이터 부족(60일 미만)으로 AI 분석이 불가능합니다.")

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
