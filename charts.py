import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def draw_chart(df, config, vol_results=None, predictions=None):
    # 인덱스 데이터 타입 보정 및 전체 날짜 정규화
    df.index = pd.to_datetime(df.index).normalize()
    view_df = df.tail(40)
    
    # 🎯 수정: 차트에 사용할 통합 x축 라벨 생성 (모든 trace에서 이 리스트를 사용해야 함)
    x_labels = view_df.index.strftime('%Y-%m-%d').tolist()
    
    target_date = config['target_date']
    target_date_ts = pd.Timestamp(target_date).normalize()
    
    if config.get('show_rsi'):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. 변동성 구름대 (Volatility Cloud)
    active_vols = config.get('vol_models', {})
    
    if vol_results and predictions and any(active_vols.values()):
        base_price_today = None
        for m, active in config.get('models', {}).items():
            if active and m in predictions:
                base_price_today = predictions[m]
                break
        if base_price_today is None:
            base_price_today = df['종가'].iloc[-1]
            
        base_price_yesterday = df['종가'].iloc[-2] if len(df) > 1 else df['종가'].iloc[-1]
        
        # 🎯 수정: x_labels에서 위치 추출
        pos_today = x_labels[-1]
        pos_yesterday = x_labels[-2] if len(x_labels) > 1 else x_labels[-1]

        settings = [
            ('egarch', 'rgba(255, 0, 0, 0.15)', 'EGARCH'),
            ('gjr_garch', 'rgba(0, 0, 255, 0.2)', 'GJR-GARCH')
        ]

        for model_key, color, label in settings:
            if active_vols.get(model_key):
                v_pct = vol_results.get(model_key, 0)
                
                # 오늘 범위
                fig.add_trace(go.Scatter(
                    x=[pos_today, pos_today], 
                    y=[base_price_today * (1 - v_pct/100), base_price_today * (1 + v_pct/100)],
                    mode='lines', name=f'{label} (오늘)',
                    line=dict(width=30, color=color),
                ), row=1, col=1)
                
                # 어제 범위
                fig.add_trace(go.Scatter(
                    x=[pos_yesterday, pos_yesterday], 
                    y=[base_price_yesterday * (1 - v_pct/100), base_price_yesterday * (1 + v_pct/100)],
                    mode='lines', name=f'{label} (어제)',
                    line=dict(width=30, color=color.replace('0.15', '0.08').replace('0.2', '0.1')),
                    showlegend=False
                ), row=1, col=1)

    # 2. 캔들스틱 (🎯 x축을 x_labels로 교체)
    fig.add_trace(go.Candlestick(
        x=x_labels, open=view_df['시가'], high=view_df['고가'], 
        low=view_df['저가'], close=view_df['종가'], name="캔들"
    ), row=1, col=1)

    # 3. RSI (🎯 x축을 x_labels로 교체)
    if config.get('show_rsi'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['RSI'], name="RSI", line=dict(color='#FFD700', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 4. 이평선 (🎯 x축을 x_labels로 교체)
    for s in config.get('ma_settings', []):
        col_n = f"{s['type']}{s['period']}"
        line_data = df['종가'].rolling(window=s['period']).mean() if s['type'] == "MA" else df['종가'].ewm(span=s['period'], adjust=False).mean()
        fig.add_trace(go.Scatter(x=x_labels, y=line_data.tail(40), name=col_n, line=dict(color=s['color'], width=s['width'])), row=1, col=1)

    # 5. 볼린저 밴드 (🎯 x축을 x_labels로 교체)
    if config.get('show_bb'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_U'], name="BB상단", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_L'], name="BB하단", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)

    # 6. 분석 범위 강조 (🎯 x0, x1을 x_labels 인덱스로 수정)
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        # view_df 내부에서의 상대적 인덱스 계산
        view_start_date = view_df.index[0]
        if target_date_ts >= view_start_date:
            rel_idx = view_df.index.get_loc(target_date_ts)
            if rel_idx > 0:
                fig.add_vrect(x0=x_labels[0], x1=x_labels[rel_idx-1], fillcolor="rgba(173, 216, 230, 0.2)", opacity=0.3, layer="below", line_width=0, row=1, col=1)

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10,r=10,t=10,b=10), showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_xaxes(type='category', tickangle=-45)
    
    return fig
