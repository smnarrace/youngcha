import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def draw_chart(df, config, vol_results=None, predictions=None):
    # 인덱스 데이터 타입 보정 및 전체 날짜 정규화
    df.index = pd.to_datetime(df.index).normalize()
    view_df = df.tail(40)
    
    target_date = config['target_date']
    target_date_ts = pd.Timestamp(target_date).normalize()
    
    if config.get('show_rsi'):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. 변동성 구름대 (Volatility Cloud) - 어제와 오늘 표시
    active_vols = config.get('vol_models', {})
    
    if vol_results and predictions and any(active_vols.values()):
        # [오늘 기준가] 선택된 모델 예측값 혹은 종가
        base_price_today = None
        for m, active in config.get('models', {}).items():
            if active and m in predictions:
                base_price_today = predictions[m]
                break
        if base_price_today is None:
            base_price_today = df['종가'].iloc[-1]
            
        # [어제 기준가] 전일 종가
        base_price_yesterday = df['종가'].iloc[-2] if len(df) > 1 else df['종가'].iloc[-1]
        
        # 차트 표시 인덱스 (마지막이 오늘, 그 앞이 어제)
        idx_today = view_df.index[-1]
        idx_yesterday = view_df.index[-2] if len(view_df) > 1 else view_df.index[-1]

        # EGARCH 구름대 (빨간색)
        if active_vols.get('egarch') and 'egarch' in vol_results:
            v_pct = vol_results['egarch']
            # 오늘 범위
            fig.add_trace(go.Scatter(
                x=[idx_today, idx_today], 
                y=[base_price_today * (1 - v_pct/100), base_price_today * (1 + v_pct/100)],
                mode='lines', name='EGARCH (오늘)',
                line=dict(width=30, color='rgba(255, 0, 0, 0.15)'),
            ), row=1, col=1)
            # 어제 범위
            fig.add_trace(go.Scatter(
                x=[idx_yesterday, idx_yesterday], 
                y=[base_price_yesterday * (1 - v_pct/100), base_price_yesterday * (1 + v_pct/100)],
                mode='lines', name='EGARCH (어제)',
                line=dict(width=30, color='rgba(255, 0, 0, 0.1)'),
                showlegend=False
            ), row=1, col=1)

        # GJR-GARCH 구름대 (파란색)
        if active_vols.get('gjr_garch') and 'gjr_garch' in vol_results:
            v_pct = vol_results['gjr_garch']
            # 오늘 범위
            fig.add_trace(go.Scatter(
                x=[idx_today, idx_today],
                y=[base_price_today * (1 - v_pct/100), base_price_today * (1 + v_pct/100)],
                mode='lines', name='GJR-GARCH (오늘)',
                line=dict(width=15, color='rgba(0, 0, 255, 0.2)'),
            ), row=1, col=1)
            # 어제 범위
            fig.add_trace(go.Scatter(
                x=[idx_yesterday, idx_yesterday],
                y=[base_price_yesterday * (1 - v_pct/100), base_price_yesterday * (1 + v_pct/100)],
                mode='lines', name='GJR-GARCH (어제)',
                line=dict(width=15, color='rgba(0, 0, 255, 0.1)'),
                showlegend=False
            ), row=1, col=1)

    # 2. 캔들스틱
    fig.add_trace(go.Candlestick(
        x=view_df.index, open=view_df['시가'], high=view_df['고가'], 
        low=view_df['저가'], close=view_df['종가'], name="캔들"
    ), row=1, col=1)

    # 3. RSI
    if config.get('show_rsi'):
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['RSI'], name="RSI", line=dict(color='#FFD700', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 4. 이평선
    for s in config.get('ma_settings', []):
        col_n = f"{s['type']}{s['period']}"
        line_data = df['종가'].rolling(window=s['period']).mean() if s['type'] == "MA" else df['종가'].ewm(span=s['period'], adjust=False).mean()
        fig.add_trace(go.Scatter(x=view_df.index, y=line_data.tail(40), name=col_n, line=dict(color=s['color'], width=s['width'])), row=1, col=1)

    # 5. 볼린저 밴드
    if config.get('show_bb'):
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['BB_U'], name="BB상단", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['BB_L'], name="BB하단", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)

    # 6. 분석 범위 강조
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        if target_idx > 0:
            fig.add_vrect(x0=view_df.index[0], x1=df.index[target_idx-1], fillcolor="rgba(173, 216, 230, 0.2)", opacity=0.3, layer="below", line_width=0, row=1, col=1)

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10,r=10,t=10,b=10), showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_xaxes(type='category', tickangle=-45)
    
    return fig
