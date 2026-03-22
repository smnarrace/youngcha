import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
import streamlit as st

def draw_chart(df, config, vol_results=None, predictions=None):

    df.index = pd.to_datetime(df.index).normalize()
    view_df = df.tail(40)
    
    x_labels = view_df.index.strftime('%Y-%m-%d').tolist()
    last_date = view_df.index[-1]
    next_date = last_date + datetime.timedelta(days=1)
    next_date_str = next_date.strftime('%Y-%m-%d')

    extended_x_labels = x_labels + [next_date_str]
 
    pos_tomorrow = next_date_str  
    pos_today = x_labels[-1]      
    
    if config.get('show_rsi'):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)

    active_vols = config.get('vol_models', {})
    if vol_results and predictions and any(active_vols.values()):
        base_price_tomorrow = None
        for m, active in config.get('models', {}).items():
            if active and m in predictions:
                base_price_tomorrow = predictions[m]
                break
        if base_price_tomorrow is None: base_price_tomorrow = df['종가'].iloc[-1]
        base_price_today_overlay = df['종가'].iloc[-2] if len(df) > 1 else df['종가'].iloc[-1]

        settings = [('egarch', 'rgba(255, 0, 0, 0.15)', 'EGARCH'), ('gjr_garch', 'rgba(0, 0, 255, 0.2)', 'GJR-GARCH')]
        for model_key, color, label in settings:
            if active_vols.get(model_key):
                v_pct = vol_results.get(model_key, 0)
                fig.add_trace(go.Bar(x=[pos_tomorrow], y=[(base_price_tomorrow * (v_pct/100)) * 2], base=base_price_tomorrow * (1 - v_pct/100), name=f'{label} (내일예측)', marker=dict(color=color.replace('0.15', '0.6').replace('0.2', '0.6'), line=dict(width=0)), width=0.8, offsetgroup=model_key), row=1, col=1)
                fig.add_trace(go.Bar(x=[pos_today], y=[(base_price_today_overlay * (v_pct/100)) * 2], base=base_price_today_overlay * (1 - v_pct/100), name=f'{label} (어제예측)', marker=dict(color=color, line=dict(width=0)), width=0.8, offsetgroup=model_key, showlegend=False), row=1, col=1)

    model_styles = {'rk4': {'color': '#FFD700', 'symbol': 'diamond'}, 'euler': {'color': '#00FF00', 'symbol': 'circle'}, 'newton': {'color': '#FF00FF', 'symbol': 'x'}, 'simpson': {'color': '#00FFFF', 'symbol': 'star'}}
    for m_key, m_val in predictions.items():
        if config['models'].get(m_key):
            style = model_styles.get(m_key, {'color': 'white', 'symbol': 'circle'})
            fig.add_trace(go.Scatter(x=[pos_tomorrow], y=[m_val], mode='markers+text', name=f'{m_key.upper()} 예측가', marker=dict(size=12, color=style['color'], symbol=style['symbol'], line=dict(width=1.5, color='white')), text=[f"{m_val:,.0f}"], textposition="top center", textfont=dict(color=style['color'], size=11)), row=1, col=1)

    fig.add_trace(go.Candlestick(x=x_labels, open=view_df['시가'], high=view_df['고가'], low=view_df['저가'], close=view_df['종가'], name="캔들"), row=1, col=1)

    if config.get('show_rsi'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['RSI'], name="RSI", line=dict(color='#FFD700', width=1.5)), row=2, col=1)
    
    for s in config.get('ma_settings', []):
        line_data = df['종가'].rolling(window=s['period']).mean() if s['type'] == "MA" else df['종가'].ewm(span=s['period'], adjust=False).mean()
        fig.add_trace(go.Scatter(x=x_labels, y=line_data.tail(40), name=f"{s['type']}{s['period']}", line=dict(color=s['color'], width=s['width'])), row=1, col=1)

    if config.get('show_bb'):
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_U'], name="BB상단", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=view_df['BB_L'], name="BB하단", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)

    target_date_obj = config.get('target_date')
    if target_date_obj:
        target_date_str = target_date_obj.strftime('%Y-%m-%d')
        if target_date_str in x_labels:
            target_idx = x_labels.index(target_date_str)
            if target_idx > 0:
                fig.add_vrect(x0=x_labels[0], x1=x_labels[target_idx-1], fillcolor="rgba(173, 216, 230, 0.2)", opacity=0.3, layer="below", line_width=0, row=1, col=1)

    if 'history' in st.session_state and len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df['date_str'] = pd.to_datetime(hist_df['date']).dt.strftime('%Y-%m-%d')
        plot_hist = hist_df[hist_df['date_str'].isin(extended_x_labels)]
        
        if 'is_buy' in plot_hist.columns:
            buy_df = plot_hist[plot_hist['is_buy'] == True]
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df['date_str'], y=buy_df['pred'],
                    mode='markers', name="AI 매수 진입",
                    marker=dict(size=12, color='#00C805', symbol='star', line=dict(width=1, color='white'))
                ), row=1, col=1)


            watch_df = plot_hist[plot_hist['is_buy'] == False]
            if not watch_df.empty:
                fig.add_trace(go.Scatter(
                    x=watch_df['date_str'], y=watch_df['pred'],
                    mode='markers', name="관망 (리스크 관리)",
                    marker=dict(size=7, color='rgba(150, 150, 150, 0.5)', symbol='x')
                ), row=1, col=1)
        else:

            fig.add_trace(go.Scatter(
                x=plot_hist['date_str'], y=plot_hist['pred'],
                mode='markers', name="예측 (데이터 갱신 필요)",
                marker=dict(size=8, color='#FF4B4B', symbol='x')
            ), row=1, col=1)


        fig.add_trace(go.Scatter(
            x=plot_hist['date_str'], y=plot_hist['pred'],
            mode='lines', name="예측 추세",
            line=dict(color='rgba(255, 75, 75, 0.2)', dash='dot', width=1),
            showlegend=False
        ), row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=600, margin=dict(l=10,r=10,t=10,b=10),
        barmode='overlay', showlegend=True, xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', categoryorder='array', categoryarray=extended_x_labels, tickangle=-45)
    )
    
    return fig
