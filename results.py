import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 하이브리드 모델 입력 데이터 변환 (7개 피처 & % 변환 적용)
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    # [전략 2 기초] 필요한 컬럼이 없으면 즉석 생성
    if '등락률' not in df.columns:
        df['등락률'] = df['종가'].pct_change() * 100
    if '거래량_변동률' not in df.columns:
        df['거래량_변동률'] = df['거래량'].pct_change() * 100
    
    # 데이터 부족 시 차단
    if target_idx < window_size:
        return None, None
        
    try:
        # A. LSTM용: 등락률 시퀀스 (이미 등락률 기반)
        seq_data = df.iloc[target_idx - window_size + 1 : target_idx + 1]['등락률'].values
        X_seq = np.array(seq_data, dtype=np.float32).reshape(1, window_size, 1)
        
        # B. [전략 1] 수치 모델 % 변환 함수
        # 공식: ((예측값 - 현재가) / 현재가) * 100
        curr_p = df.iloc[target_idx]['종가']
        def to_pct(val, base):
            return ((val - base) / base) * 100 if base != 0 else 0

        # C. [전략 2] 7개 피처 구성 (수치3 + 변동성2 + RSI + 거래량)
        X_static = np.array([[
            to_pct(predictions.get('euler', curr_p), curr_p),   # 오일러 %
            to_pct(predictions.get('rk4', curr_p), curr_p),     # RK4 %
            to_pct(predictions.get('newton', curr_p), curr_p),  # 뉴턴 %
            float(vol_results.get('egarch', 0)),              # EGARCH
            float(vol_results.get('gjr_garch', 0)),           # GJR-GARCH
            float(df.iloc[target_idx].get('RSI', 50)),         # RSI (심리)
            float(df.iloc[target_idx].get('거래량_변동률', 0))   # 거래량 에너지
        ]], dtype=np.float32)
        
        return np.nan_to_num(X_seq), np.nan_to_num(X_static)
    except Exception as e:
        st.error(f"데이터 준비 중 오류: {e}")
        return None, None

# 2. 메인 결과 렌더링 함수
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 AI 분석 결 ")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    target_idx = df.index.get_loc(target_date_ts)
    prev_price = df.iloc[target_idx]['종가'] # 현재 시점의 종가 (예측의 기준)

    # --- AI 예측 프로세스 ---
    if config['models'].get('hybrid') and 'hybrid_model' in st.session_state:
        X_seq, X_static = prepare_hybrid_input(df, target_idx, vol_results, predictions)
        
        if X_seq is not None:
            # 모델 예측 (7개 피처를 받아 등락률 %를 반환)
            pred_res = st.session_state.hybrid_model.predict(X_seq, X_static)
            
            if pred_res is not None and len(pred_res) > 0:
                pred_pct = pred_res[0] 
                # [복구] 예측 등락률을 다시 가격으로 환산
                hybrid_pred_val = prev_price * (1 + pred_pct / 100)
                
                # 결과 지표 출력
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("AI 예측 등락", f"{pred_pct:+.2f}%")
                with c2:
                    st.metric("AI 목표가", f"{int(hybrid_pred_val):,}원")
                with c3:
                    # 다음 거래일 데이터가 있다면 실제 오차율 계산
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
                st.error("❌ 모델이 예측값을 생성하지 못했습니다. 재학습이 필요합니다.")
        else:
            st.info("💡 과거 데이터 부족으로 분석을 수행할 수 없습니다.")

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
        # 실제가 vs 예측가 추세 차트
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['actual'], name="실제가", line=dict(color="#1C83E1")))
        fig_trend.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['pred'], name="AI예측", line=dict(color="#FF4B4B", dash='dot')))
        fig_trend.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig_trend, use_container_width=True)
