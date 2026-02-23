import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def draw_chart(df, config, vol_results=None, predictions=None):
    df.index = pd.to_datetime(df.index)
    view_df = df.tail(40)
    target_date = config['target_date']
    target_date_ts = pd.Timestamp(target_date).normalize()
    
    if config['show_rsi']:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. 캔들스틱
    fig.add_trace(go.Candlestick(
        x=view_df.index, open=view_df['시가'], high=view_df['고가'], 
        low=view_df['저가'], close=view_df['종가'], name="캔들"
    ), row=1, col=1)

    # 2. 변동성 구름대 (Volatility Cloud)
    # config['vol_models']가 있는지 확인하여 에러 방지
    active_vols = config.get('vol_models', {})
    
    if vol_results and predictions and any(active_vols.values()):
        # 기준 가격 설정: 선택된 수치해석 모델 중 첫 번째 것을 쓰거나, 없으면 현재가 사용
        base_price = None
        for m, active in config['models'].items():
            if active and m in predictions:
                base_price = predictions[m]
                break
        
        if base_price is None:
            base_price = df['종가'].iloc[-1]
            
        # EGARCH 구름대
        if active_vols.get('egarch') and 'egarch' in vol_results:
            v_pct = vol_results['egarch']
            upper = base_price * (1 + v_pct/100)
            lower = base_price * (1 - v_pct/100)
            
            fig.add_trace(go.Scatter(
                x=[view_df.index[-1], view_df.index[-1]], 
                y=[lower, upper],
                mode='lines',
                name='EGARCH 범위',
                line=dict(width=30, color='rgba(255, 0, 0, 0.15)'),
            ), row=1, col=1)

        # GJR-GARCH 구름대
        if active_vols.get('gjr_garch') and 'gjr_garch' in vol_results:
            v_pct = vol_results['gjr_garch']
            upper = base_price * (1 + v_pct/100)
            lower = base_price * (1 - v_pct/100)
            
            fig.add_trace(go.Scatter(
                x=[view_df.index[-1], view_df.index[-1]],
                y=[lower, upper],
                mode='lines',
                name='GJR-GARCH 범위',
                line=dict(width=15, color='rgba(0, 0, 255, 0.2)'),
            ), row=1, col=1)

    # 3. RSI
    if config['show_rsi']:
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['RSI'], name="RSI", line=dict(color='#FFD700', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 4. 이평선
    for s in config['ma_settings']:
        col_n = f"{s['type']}{s['period']}"
        line_data = df['종가'].rolling(window=s['period']).mean() if s['type'] == "MA" else df['종가'].ewm(span=s['period'], adjust=False).mean()
        fig.add_trace(go.Scatter(x=view_df.index, y=line_data.tail(40), name=col_n, line=dict(color=s['color'], width=s['width'])), row=1, col=1)

    # 5. 볼린저 밴드
    if config['show_bb']:
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['BB_U'], name="BB상단", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['BB_L'], name="BB하단", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)

    # 6. 분석 범위 강조
    # df.index가 datetime이므로 target_date_ts와 정확히 비교됨
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        if target_idx > 0:
            fig.add_vrect(x0=view_df.index[0], x1=df.index[target_idx-1], fillcolor="rgba(173, 216, 230, 0.2)", opacity=0.3, layer="below", line_width=0, annotation_text="분석 범위", row=1, col=1)

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10,r=10,t=10,b=10), showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_xaxes(type='category', tickangle=-45)
    return fig
이 chart.py 코드에서 YYYY-MM-DD로 출력되게끔 수정은 어떻게해야해
