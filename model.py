import numpy as np
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class YoungChaHybridModel:
    def __init__(self):
        self.lstm_model = None
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        self.is_trained = False

    def train(self, X_seq, X_static, y):
        # [★핵심 방어선] 텐서플로우가 에러를 뿜지 못하도록 완벽한 실수형 배열로 강제 변환
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # 1. LSTM 모델 구성
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dense(16, activation='relu'),
            Dense(1) # 최종 출력 1개
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # 이제 에러 없이 부드럽게 학습됩니다!
        self.lstm_model.fit(X_seq, y, epochs=20, verbose=0)
        
        # 2. LSTM 특징 추출 및 XGBoost 학습
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        self.xgb_model.fit(combined_X, y)
        self.is_trained = True

    def predict(self, X_seq, X_static):
        if not self.is_trained:
            return None 
            
        # 예측할 때도 동일하게 형변환 안전장치 적용
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
            
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        return self.xgb_model.predict(combined_X)
