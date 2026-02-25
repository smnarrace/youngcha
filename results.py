import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# 하이브리드 모델 입력 데이터 변환 함수
def prepare_hybrid_input(df, target_idx, vol_results, predictions, window_size=60):
    # 1. 시계열 데이터 변환 (LSTM용)
    # target_idx 직전까지의 데이터를 window_size만큼 추출
    if target_idx < window_size:
        return None, None
        
    seq_data = df.iloc[target_idx - window_size : target_idx]['종가'].values
    X_seq = seq_data.reshape(1, window_size, 1) # (샘플수, 타임스텝, 피처수)
    
    # 2. 정적 수치 데이터 변환 (XGBoost용)
    # 수치해석 결과와 GARCH 변동성 결과를 하나의 행으로 결합
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
    # 사용자가 하나라도 체크했을 때만 섹션 표시
    if vol_results and any(config['vol_models'].values()):
        st.markdown("### 📉 변동성 분석 (Volatility Models)")
        v_col1, v_col2 = st.columns(2)
        
        # EGARCH 체크 시 표시
        if config['vol_models'].get('egarch'):
            with v_col1:
                st.metric("EGARCH 변동성", f"{vol_results['egarch']:.2f}%")
                st.caption("하락장에 민감한 변동성 지수")
        
        # GJR-GARCH 체크 시 표시
        if config['vol_models'].get('gjr_garch'):
            with v_col2:
                st.metric("GJR-GARCH 변동성", f"{vol_results['gjr_garch']:.2f}%")
                st.caption("상승/하락 비대칭 리스크")
        st.divider()

    # --- [NEW] 하이브리드 모델 예측 실행 ---
    hybrid_pred_val = None
    # 사이드바에서 hybrid 모델이 체크되었고, 관련 데이터가 모두 있을 때 실행
    if config['models'].get('hybrid') and vol_results and predictions:
        X_seq, X_static = prepare_hybrid_input(df, target_idx, vol_results, predictions)
        
        if X_seq is not None and 'hybrid_model' in st.session_state:
            # st.session_state에 저장된 모델로 예측 수행
            hybrid_pred_val = st.session_state.hybrid_model.predict(X_seq, X_static)[0]
        elif X_seq is None:
            st.warning("⚠️ 하이브리드 모델을 실행하기 위한 과거 데이터(최소 60일)가 부족합니다.")

    # --- 2. 예측 모델 비교 섹션 ---
    if predictions:
        st.markdown("### 🧮 예측 모델 비교")
        comparison_data = []

        # (1) 하이브리드 모델 결과 먼저 추가
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

        # (2) 기존 수치해석 모델 결과 루프
        for m, active in config['models'].items():
            if not active or m == 'hybrid': continue # hybrid는 위에서 처리했으므로 패스
            
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
                is_hit = (pred_val > prev_price and actual_price > prev_price) or \
                         (pred_val < prev_price and actual_price < prev_price)
                
                comparison_data.append({
                    "모델": m.upper(), 
                    "예측가": f"{int(pred_val):,}원", 
                    "실제가": f"{int(actual_price):,}원",
                    "오차율": f"{error_rate:+.2f}%", 
                    "방향": "🟢 적중" if is_hit else "🔴 실패"
                })

        # 테이블 렌더링
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Pandas 버전에 따라 applymap 또는 map 사용 (최신 버전은 map 권장)
            styled_df = comp_df.style.map(
                lambda v: 'color: #00C805; font-weight: bold' if '🟢' in str(v) else 
                          ('color: #FF4B4B; font-weight: bold' if '🔴' in str(v) else ''), 
                subset=['방향']
            )
            st.table(styled_df)
            st.write(f"**전일 대비 변화:** {int(prev_price):,}원 → {int(actual_price):,}원")
        else:
            st.info("💡 왼쪽 사이드바에서 예측 모델을 선택해 주세요.")
