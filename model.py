import numpy as np
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class YoungChaHybridModel:
    def __init__(self):
        self.lstm_model = None
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05)

    def train(self, X_seq, X_static, y):
        # 1. LSTM으로 시계열 패턴 학습 (종가 흐름 등)
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dense(16)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X_seq, y, epochs=20, verbose=0)
        
        # 2. LSTM에서 특징 추출
        lstm_feats = self.lstm_model.predict(X_seq)
        
        # 3. XGBoost 학습 (LSTM 특징 + GARCH 변동성 + 수치해석 결과)
        combined_X = np.hstack([X_static, lstm_feats])
        self.xgb_model.fit(combined_X, y)

    def predict(self, X_seq, X_static):
        lstm_feats = self.lstm_model.predict(X_seq)
        combined_X = np.hstack([X_static, lstm_feats])
        return self.xgb_model.predict(combined_X)
