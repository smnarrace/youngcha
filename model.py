import numpy as np
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

class YoungChaHybridModel:
    def __init__(self):
        self.lstm_model = None
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8)
        self.is_trained = False

    def train(self, X_seq, X_static, y):
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # [★핵심 수정] LSTM에게 줄 정답(y)의 차원을 (데이터개수, 1)로 세로로 세워줍니다!
        y_lstm = y.reshape(-1, 1)

        # 1. LSTM 모델 구성
        self.lstm_model = Sequential([
            Input(shape=(X_seq.shape[1], X_seq.shape[2])),
            LSTM(64),
            Dense(16, activation='relu'),
            Dense(1) # 최종 출력 1개
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # [★핵심 수정] 여기서 y 대신 차원을 맞춘 y_lstm을 넣습니다.
        self.lstm_model.fit(X_seq, y_lstm, epochs=20, verbose=0)
        
        # 2. LSTM 특징 추출 및 XGBoost 학습
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        
        # XGBoost는 원래대로 1차원 y를 줘도 알아서 잘 소화합니다.
        self.xgb_model.fit(combined_X, y)
        self.is_trained = True

    def predict(self, X_seq, X_static):
        if not self.is_trained:
            return None 
            
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
            
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        return self.xgb_model.predict(combined_X)
