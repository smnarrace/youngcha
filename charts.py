import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime

def draw_chart(df, config, vol_results=None, predictions=None):
    # 인덱스 데이터 타입 보정 및 전체 날짜 정규화
    df.index = pd.to_datetime(df.index).normalize()
    view_df = df.tail(40)
    
    # 🎯 1. 미래 날짜(내일) 한 칸 추가 로직
    x_labels = view_df.index.strftime('%Y-%m-%d').tolist()
    last_date = view_df.index[-1]
    next_date = last_date + datetime.timedelta(days=1)
    next_date_str = next_date.strftime('%Y-%m-%d')
    
    # 전체 X축 라벨: 기존 40개 + 미래 1개
    extended_x_labels = x_labels + [next_date_str]
    
    # 위치 고정
    pos_tomorrow = next_date_str  # 오늘의 예측이 표시될 내일 칸
    pos_today = x_labels[-1]      # 어제의 예측이 표시될 오늘 캔들 칸
    
    if config.get('show_rsi'):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)

    # 2. 변동성 구름대 (go.Bar 방식)
    active_vols = config.get('vol_models', {})
    
    if vol_results and predictions and any(active_vols.values()):
        # [오늘의 예측 기준가] -> 내일 위치용 (수치해석 모델 결과값)
        base_price_tomorrow = None
        for m, active in config.get('models', {}).items():
            if active and m in predictions:
                base_price_tomorrow = predictions[m]
                break
        if base_price_tomorrow is None:
            base_price_tomorrow = df['종가'].iloc[-1]
            
        # [어제의 예측 기준가] -> 오늘 위치용 (전일 종가 기준)
        # 🎯 어제 시점의 예측은 어제 종가를 기준으로 해야 오늘 캔들과 정확히 일치합니다.
        base_price_today_overlay = df['종가'].iloc[-2] if len(df) > 1 else df['종가'].iloc[-1]

        settings = [
            ('egarch', 'rgba(255, 0, 0, 0.15)', 'EGARCH'),
            ('gjr_garch', 'rgba(0, 0, 255, 0.2)', 'GJR-GARCH')
        ]

        for model_key, color, label in settings:
            if active_vols.get(model_key):
                v_pct = vol_results.get(model_key, 0)
                
                # 🔥 오늘의 예측 (내일 빈 칸 - 윤곽선 없고 진하게)
                fig.add_trace(go.Bar(
                    x=[pos_tomorrow], 
                    y=[(base_price_tomorrow * (v_pct/100)) * 2], 
                    base=base_price_tomorrow * (1 - v_pct/100), 
                    name=f'{label} (내일예측)',
                    marker=dict(
                        color=color.replace('0.15', '0.6').replace('0.2', '0.6'), 
                        line=dict(width=0)
                    ),
                    width=0.8,
                    offsetgroup=model_key,
                ), row=1, col=1)
                
                # 🔥 어제의 예측 (오늘 캔들 오버레이 - 윤곽선 없고 연하게)
                fig.add_trace(go.Bar(
                    x=[pos_today], 
                    y=[(base_price_today_overlay * (v_pct/100)) * 2],
                    base=base_price_today_overlay * (1 - v_pct/100),
                    name=f'{label} (어제예측)',
                    marker=dict(color=color, line=dict(width=0)),
                    width=0.8,
                    offsetgroup=model_key,
                    showlegend=False
                ), row=1, col=1)

        # 🎯 모델별 고유 컬러 및 스타일 설정
        model_styles = {
            'rk4': {'color': '#FFD700', 'symbol': 'diamond'},      # 골드 / 다이아몬드 (가장 정교)
            'euler': {'color': '#00FF00', 'symbol': 'circle'},     # 라임 / 원 (기본)
            'newton': {'color': '#FF00FF', 'symbol': 'x'},          # 마젠타 / X
            'simpson': {'color': '#00FFFF', 'symbol': 'star'}      # 사이언 / 별
        }

        # 🎯 수치해석 모델 예측 점 표시 (내일 칸 중앙)
        for m_key, m_val in predictions.items():
            if config['models'].get(m_key):
                style = model_styles.get(m_key, {'color': 'white', 'symbol': 'circle'})
                
                fig.add_trace(go.Scatter(
                    x=[pos_tomorrow], 
                    y=[m_val],
                    mode='markers+text',
                    name=f'{m_key.upper()} 예측가',
                    marker=dict(
                        size=12, 
                        color=style['color'], 
                        symbol=style['symbol'], 
                        line=dict(width=1.5, color='white') # 가독성을 위한 테두리
                    ),
                    text=[f"{m_val:,.0f}"], 
                    textposition="top center",
                    textfont=dict(color=style['color'], size=11) # 텍스트 색상도 모델색과 일치
                ), row=1, col=1)

    # 3. 캔들스틱
    fig.add_trace(go.Candlestick(
        x=x_labels, open=view_df['시가'], high=view_df['고가'], 
        low=view_df['저가'], close=view_df['종가'], name="캔들"
    ), row=1, col=1)

    # ... (RSI, 이평선, 볼린저 밴드 로직 동일) ...
    if config.get('show_rsi'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['RSI'], name="RSI", line=dict(color='#FFD700', width=1.5)), row=2, col=1)
    
    for s in config.get('ma_settings', []):
        line_data = df['종가'].rolling(window=s['period']).mean() if s['type'] == "MA" else df['종가'].ewm(span=s['period'], adjust=False).mean()
        fig.add_trace(go.Scatter(x=x_labels, y=line_data.tail(40), name=f"{s['type']}{s['period']}", line=dict(color=s['color'], width=s['width'])), row=1, col=1)

    if config.get('show_bb'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_U'], name="BB상단", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_L'], name="BB하단", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)

    # 7. 분석 범위 강조
    target_date_obj = config.get('target_date')
    if target_date_obj:
        target_date_str = target_date_obj.strftime('%Y-%m-%d')
        if target_date_str in x_labels:
            target_idx = x_labels.index(target_date_str)
            if target_idx > 0:
                fig.add_vrect(x0=x_labels[0], x1=x_labels[target_idx-1], fillcolor="rgba(173, 216, 230, 0.2)", opacity=0.3, layer="below", line_width=0, row=1, col=1)

    # 8. 레이아웃 최종 업데이트
    fig.update_layout(
        template="plotly_dark", height=600, margin=dict(l=10,r=10,t=10,b=10),
        barmode='overlay', showlegend=True, xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', categoryorder='array', categoryarray=extended_x_labels, tickangle=-45)
    )
    
    return fig
