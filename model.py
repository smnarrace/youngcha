import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, GlobalAveragePooling1D

class YoungChaHybridModel:
    def __init__(self):
        self.lstm_model = None
        # XGBoost 하이퍼파라미터 살짝 강화
        self.xgb_model = XGBRegressor(
            n_estimators=150, 
            learning_rate=0.03, 
            max_depth=4, 
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.is_trained = False

    def build_attention_lstm(self, seq_shape):
        """어텐션이 적용된 LSTM 모델 설계"""
        inputs = Input(shape=seq_shape)
        # return_sequences=True여야 어텐션이 각 시점의 가중치를 계산할 수 있습니다.
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        lstm_out = LSTM(32, return_sequences=True)(lstm_out)
        
        # --- 어텐션 층 ---
        query_value = lstm_out
        attention_out = Attention()([query_value, query_value])
        
        # 어텐션 결과물을 압축하여 특징 벡터 생성
        avg_pool = GlobalAveragePooling1D()(attention_out)
        dense_out = Dense(16, activation='relu')(avg_pool)
        
        model = Model(inputs=inputs, outputs=dense_out)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_seq, X_static, y_actual, y_numerical_base):
        """
        y_actual: 실제 주가 등락률
        y_numerical_base: (오일러 + RK4 + 뉴턴) / 3 의 평균 등락률
        """
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
        
        # [핵심] 잔차(Residual) 계산: 실제값 - 수치해석 평균값
        y_residual = np.asarray(y_actual - y_numerical_base, dtype=np.float32)

        # 1. 어텐션 LSTM 학습
        if self.lstm_model is None:
            self.lstm_model = self.build_attention_lstm((X_seq.shape[1], X_seq.shape[2]))
        
        # LSTM은 이 잔차(오차)의 패턴을 학습합니다.
        self.lstm_model.fit(X_seq, y_residual, epochs=30, verbose=0, batch_size=32)

        # 2. 특징 추출 및 XGBoost 학습
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        
        # XGBoost가 수치모델들의 결과와 AI의 보정치를 최종 결합하여 잔차를 예측합니다.
        self.xgb_model.fit(combined_X, y_residual)
        self.is_trained = True

    def predict(self, X_seq, X_static):
        if not self.is_trained:
            return None
            
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
            
        lstm_feats = self.lstm_model.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        
        # AI가 생각하는 '수치해석 모델의 오차 보정값'을 반환합니다.
        return self.xgb_model.predict(combined_X)
