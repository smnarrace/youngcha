import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def draw_chart(df, config, vol_results=None, predictions=None):
    df.index = pd.to_datetime(df.index)
    view_df = df.tail(40)
    
    # --- [추가] x축에 표시할 깔끔한 날짜 라벨 생성 ---
    clean_date_labels = [d.strftime('%Y-%m-%d') for d in view_df.index]
    # ----------------------------------------------

    target_date = config['target_date']
    target_date_ts = pd.Timestamp(target_date).normalize()
    
    if config['show_rsi']:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. 캔들스틱 (x축 데이터는 그대로 view_df.index 사용)
    fig.add_trace(go.Candlestick(
        x=view_df.index, open=view_df['시가'], high=view_df['고가'], 
        low=view_df['저가'], close=view_df['종가'], name="캔들"
    ), row=1, col=1)

    # ... [중간 변동성 모델 및 보조지표 로직 동일] ...

    # 6. 분석 범위 강조
    if target_date_ts in df.index:
        target_idx = df.index.get_loc(target_date_ts)
        if target_idx > 0:
            fig.add_vrect(x0=view_df.index[0], x1=df.index[target_idx-1], fillcolor="rgba(173, 216, 230, 0.2)", opacity=0.3, layer="below", line_width=0, annotation_text="분석 범위", row=1, col=1)

    # --- [수정] 레이아웃 및 x축 설정 ---
    fig.update_layout(
        template="plotly_dark", 
        height=600, 
        margin=dict(l=10, r=10, t=10, b=10), 
        showlegend=True, 
        xaxis_rangeslider_visible=False
    )

    # 메인 차트(row=1) x축 설정
    fig.update_xaxes(
        type='category', 
        tickmode='array',
        tickvals=view_df.index,       # 실제 데이터 위치
        ticktext=clean_date_labels,   # 화면에 보여줄 텍스트 (YYYY-MM-DD)
        tickangle=-45, 
        row=1, col=1
    )
    
    # RSI 차트(row=2)가 있을 경우 동일하게 적용 (라벨은 숨김 처리 가능)
    if config['show_rsi']:
        fig.update_xaxes(
            type='category', 
            tickmode='array',
            tickvals=view_df.index,
            ticktext=clean_date_labels,
            showticklabels=False,      # 중복 방지를 위해 아래쪽은 숨김
            row=2, col=1
        )

    return fig
