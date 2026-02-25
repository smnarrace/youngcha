import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# 1. 데이터 준비 (가정: df에 주가와 기술적 지표가 들어있음)
# X_seq: LSTM용 (60일치 흐름), X_static: XGBoost용 (현재 변동성, RSI 등)
# y: 예측 대상 (내일 종가)

def build_hybrid_model(X_seq_train, X_static_train, y_train):
    
    # --- STEP 1: LSTM 모델 (시계열 특징 추출기) ---
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
        Dense(10) # 10개의 핵심 시계열 특징 추출
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_seq_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # LSTM을 통해 시계열 데이터의 '압축된 특징'을 뽑아냄
    lstm_features = lstm_model.predict(X_seq_train)
    
    # --- STEP 2: 데이터 결합 ---
    # 수치해석 지표(변동성 등) + LSTM이 찾아낸 패턴
    combined_features = np.hstack([X_static_train, lstm_features])
    
    # --- STEP 3: XGBoost 모델 (최종 의사결정자) ---
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    xgb_model.fit(combined_features, y_train)
    
    return lstm_model, xgb_model

print("하이브리드 모델 구조 생성 완료")
