import streamlit as st
import pandas as pd
# logic_calculate에서 변동성 모델 함수 추가 임포트
from logic_calculate import get_numerical_analysis, get_volatility_models

def render_results(df, config):
    st.subheader(f"🎯 {config['target_date']} 분석 결과")
    target_date_ts = pd.Timestamp(config['target_date'])
    
    if target_date_ts not in df.index:
        st.warning("⚠️ 선택한 날짜의 데이터가 존재하지 않습니다.")
        return

    # 데이터 인덱싱 및 가격 추출
    target_idx = df.index.get_loc(target_date_ts)
    actual_price = df.loc[target_date_ts, '종가']
    prev_price = df.iloc[target_idx - config['step_size']]['종가']
    past_prices = df.iloc[:target_idx]['종가'].values

    # --- [신규 추가] GARCH 변동성 분석 섹션 ---
    st.markdown("### 📉 변동성 분석 (Volatility Models)")
    # 최근 500일 데이터를 기준으로 변동성 계산
    vol_results = get_volatility_models(past_prices)
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.metric("EGARCH 변동성", f"{vol_results['egarch']:.2f}%")
        st.caption("하락장에 더 민감하게 반응한 변동성 지수입니다.")
    with v_col2:
        st.metric("GJR-GARCH 변동성", f"{vol_results['gjr_garch']:.2f}%")
        st.caption("상승/하락 비대칭성을 반영한 리스크 수치입니다.")
    st.divider()
    # ------------------------------------------

    # 기존 수치해석 모델 계산
    predictions = get_numerical_analysis(past_prices, h=config['step_size'])
    
    comparison_data = []
    for m, active in config['models'].items():
        if not active: continue
        
        pred_val = predictions[m]
        if m == "simpson":
            comparison_data.append({"모델": "SIMPSON", "예측/에너지": f"{pred_val:.4f}", "실제가": int(actual_price), "오차율": "N/A", "방향": "-"})
        else:
            error_rate = ((actual_price - pred_val) / actual_price) * 100
            is_hit = (pred_val > prev_price and actual_price > prev_price) or (pred_val < prev_price and actual_price < prev_price)
            comparison_data.append({
                "모델": m.upper(), "예측가": f"{int(pred_val):,}원", "실제가": f"{int(actual_price):,}원",
                "오차율": f"{error_rate:+.2f}%", "방향": "🟢 적중" if is_hit else "🔴 실패"
            })

    if comparison_data:
        st.markdown("### 🧮 수치해석 모델 비교")
        comp_df = pd.DataFrame(comparison_data)
        st.table(comp_df.style.applymap(lambda v: 'color: #00C805; font-weight: bold' if '🟢' in str(v) else ('color: #FF4B4B; font-weight: bold' if '🔴' in str(v) else ''), subset=['방향']))
        st.write(f"**전일 대비 변화:** {int(prev_price):,}원 → {int(actual_price):,}원")
    else:
        st.info("💡 왼쪽 사이드바에서 수치해석 모델을 선택해 주세요.")
