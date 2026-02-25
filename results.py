import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 하이브리드 모델 입력 데이터 변환 (방어 로직 강화)
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    # 등락률 데이터가 없으면 즉시 생성
    if '등락률' not in df.columns:
        df['등락률'] = df['종가'].pct_change() * 100
        df = df.fillna(0)

    # 과거 데이터 부족 시 차단
    if target_idx < window_size:
        return None, None
        
    try:
        # LSTM용: 등락률 시퀀스 (window_size만큼)
        seq_data = df.iloc[target_idx - window_size + 1 : target_idx + 1]['등락률'].values
        X_seq = np.array(seq_data, dtype=np.float32).reshape(1, window_size, 1)
        
        # XGBoost용: 수치 및 변동성 피처
        X_static = np.array([[
            float(predictions.get('euler', 0)),
            float(predictions.get('rk4', 0)),
            float(predictions.get('newton', 0)),
            float(vol_results.get('egarch', 0)),
            float(vol_results.get('gjr_garch', 0))
        ]], dtype=np.float32)
        
        # 결측치(NaN, Inf)를 0으로 완벽 제거
        return np.nan_to_num(X_seq), np.nan_to_num(X_static)
    except Exception:
        return None, None

# 2. 메인 결과 렌더링 함수
def render_results(df, config, vol_results=None, predictions=None):
    st.subheader(f"🎯 {config['target_date']} AI 분석 리포트")
    target_date_ts = pd.Timestamp(config['target_date']).normalize()
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    target_idx = df.index.get_loc(target_date_ts)
    prev_price = df.iloc[target_idx - 1]['종가']
    actual_price = df.iloc[target_idx]['종가']

    # --- AI 예측 프로세스 ---
    if config['models'].get('hybrid') and 'hybrid_model' in st.session_state:
        X_seq, X_static = prepare_hybrid_input(df, target_idx, vol_results, predictions)
        
        if X_seq is not None:
            # [수정 포인트] predict 결과가 비어있는지 확인 후 접근
            pred_res = st.session_state.hybrid_model.predict(X_seq, X_static)
            
            if pred_res is not None and len(pred_res) > 0:
                pred_pct = pred_res[0] # 모델이 뱉은 등락률(%)
                # 가격 복구: 전일종가 * (1 + 등락률/100)
                hybrid_pred_val = prev_price * (1 + pred_pct / 100)
                
                # 결과 지표 출력
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("AI 예측 등락", f"{pred_pct:+.2f}%")
                with c2:
                    st.metric("AI 예측가", f"{int(hybrid_pred_val):,}원")
                with c3:
                    error_rate = ((actual_price - hybrid_pred_val) / actual_price) * 100
                    st.metric("오차율", f"{error_rate:+.2f}%")
                
                # 성능 검증 섹션 (게이지 등)
                if 'history' in st.session_state and len(st.session_state.history) > 0:
                    st.divider()
                    render_performance_visuals()
            else:
                st.error("❌ 모델이 예측값을 생성하지 못했습니다. 다시 학습시켜 주세요.")
        else:
            st.info("💡 과거 데이터 부족으로 분석을 수행할 수 없습니다.")

def render_performance_visuals():
    # 세션 히스토리를 이용한 게이지 및 차트 출력 로직
    hist_df = pd.DataFrame(st.session_state.history).sort_values("date")
    hit_rate = (hist_df['hit'].sum() / len(hist_df)) * 100
    
    col_g, col_t = st.columns([1, 2])
    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = hit_rate,
            title = {'text': "방향 적중률", 'font': {'size': 16}},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00C805"}}
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_t:
        st.line_chart(hist_df.set_index('date')[['actual', 'pred']])
