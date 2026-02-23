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

    # 2. 변동성 구름대 (Volatility Cloud)
    active_vols = config.get('vol_models', {})
    
    if vol_results and predictions and any(active_vols.values()):
        # [오늘의 예측 기준가] -> 내일 위치(pos_tomorrow)에 표시
        base_price_today = None
        for m, active in config.get('models', {}).items():
            if active and m in predictions:
                base_price_today = predictions[m]
                break
        if base_price_today is None:
            base_price_today = df['종가'].iloc[-1]
            
        # [어제의 예측 기준가] -> 오늘 위치(pos_today)에 표시
        base_price_yesterday = df['종가'].iloc[-2] if len(df) > 1 else df['종가'].iloc[-1]

        settings = [
            ('egarch', 'rgba(255, 0, 0, 0.15)', 'EGARCH'),
            ('gjr_garch', 'rgba(0, 0, 255, 0.2)', 'GJR-GARCH')
        ]

        for model_key, color, label in settings:
            if active_vols.get(model_key):
                v_pct = vol_results.get(model_key, 0)
                
                # 🔥 오늘의 예측 (내일 빈 칸에 표시)
                fig.add_trace(go.Scatter(
                    x=[pos_tomorrow, pos_tomorrow], 
                    y=[base_price_today * (1 - v_pct/100), base_price_today * (1 + v_pct/100)],
                    mode='lines', name=f'{label} (내일예측)',
                    line=dict(width=30, color=color),
                ), row=1, col=1)
                
                # 🔥 어제의 예측 (오늘 캔들과 겹치게 표시)
                fig.add_trace(go.Scatter(
                    x=[pos_today, pos_today], 
                    y=[base_price_yesterday * (1 - v_pct/100), base_price_yesterday * (1 + v_pct/100)],
                    mode='lines', name=f'{label} (어제예측)',
                    line=dict(width=30, color=color.replace('0.15', '0.08').replace('0.2', '0.1')),
                    showlegend=False
                ), row=1, col=1)

    # 3. 캔들스틱
    fig.add_trace(go.Candlestick(
        x=x_labels, open=view_df['시가'], high=view_df['고가'], 
        low=view_df['저가'], close=view_df['종가'], name="캔들"
    ), row=1, col=1)

    # 4. RSI
    if config.get('show_rsi'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['RSI'], name="RSI", line=dict(color='#FFD700', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 5. 이평선
    for s in config.get('ma_settings', []):
        col_n = f"{s['type']}{s['period']}"
        line_data = df['종가'].rolling(window=s['period']).mean() if s['type'] == "MA" else df['종가'].ewm(span=s['period'], adjust=False).mean()
        fig.add_trace(go.Scatter(x=x_labels, y=line_data.tail(40), name=col_n, line=dict(color=s['color'], width=s['width'])), row=1, col=1)

    # 6. 볼린저 밴드
    if config.get('show_bb'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_U'], name="BB상단", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_L'], name="BB하단", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)

    # 7. 분석 범위 강조 (target_date_ts 정의 및 에러 방지 처리)
    target_date_obj = config.get('target_date')
    if target_date_obj:
        target_date_str = target_date_obj.strftime('%Y-%m-%d')
        if target_date_str in x_labels:
            target_idx = x_labels.index(target_date_str)
            if target_idx > 0:
                fig.add_vrect(
                    x0=x_labels[0], 
                    x1=x_labels[target_idx-1], 
                    fillcolor="rgba(173, 216, 230, 0.2)", 
                    opacity=0.3, 
                    layer="below", 
                    line_width=0, 
                    row=1, col=1
                )

    # 8. 레이아웃 최종 업데이트
    fig.update_layout(
        template="plotly_dark", 
        height=600, 
        margin=dict(l=10,r=10,t=10,b=10), 
        showlegend=True, 
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=extended_x_labels,
            tickangle=-45
        )
    )
    
    return fig
