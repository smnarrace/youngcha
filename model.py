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
        # 1. LSTM으로 시계열 패턴 학습
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dense(16, activation='relu'), # 16개의 숨겨진 패턴을 찾고
            Dense(1)                      # [핵심 수정] 최종 출력을 정답(y)과 똑같이 1개로 맞춤!
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # 이제 에러 없이 부드럽게 학습됩니다.
        self.lstm_model.fit(X_seq, y, epochs=20, verbose=0)
        
        # 2. LSTM에서 특징 추출 (여기서는 LSTM이 1차로 예측한 값을 피처로 사용)
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        
        # 3. XGBoost 학습 (수치해석/GARCH 데이터 + LSTM 예측값 결합)
        combined_X = np.hstack([X_static, lstm_feats])
        self.xgb_model.fit(combined_X, y)
        
        self.is_trained = True

    def predict(self, X_seq, X_static):
        if not self.is_trained:
            return None 
            
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        return self.xgb_model.predict(combined_X)
