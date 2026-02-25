import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, GlobalAveragePooling1D

class YoungChaHybridModel:
    def __init__(self):
        self.lstm_model = None
        self.feature_extractor = None # 특징만 뽑아내는 보조 모델
        
        self.xgb_model = XGBRegressor(
            n_estimators=150, 
            learning_rate=0.03, 
            max_depth=4, 
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.is_trained = False

    def build_attention_lstm(self, seq_shape):
        inputs = Input(shape=seq_shape)
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        lstm_out = LSTM(32, return_sequences=True)(lstm_out)
        
        # 어텐션 층
        attention_out = Attention()([lstm_out, lstm_out])
        avg_pool = GlobalAveragePooling1D()(attention_out)
        
        # 특징 벡터 (나중에 XGBoost에게 넘겨줄 요약본)
        features = Dense(16, activation='relu', name='features')(avg_pool)
        
        # [핵심] 3대 수치모델(Euler, RK4, Newton)의 가중치를 결정하는 레이어 (합이 1이 됨)
        weights_out = Dense(3, activation='softmax', name='weights')(features)
        
        # 1. 가중치 학습용 전체 모델
        model = Model(inputs=inputs, outputs=weights_out)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # 2. 특징 추출용 보조 모델
        feature_extractor = Model(inputs=inputs, outputs=features)
        
        return model, feature_extractor

    def train(self, X_seq, X_static, y_actual, y_numerical_base):
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
        y_actual = np.asarray(y_actual, dtype=np.float32)

        # ----------------------------------------------------
        # 🎯 [1단계] "누가 더 잘 맞췄나?" -> 완벽한 가중치 정답지 만들기
        # X_static 의 1, 2, 3번째 값이 각각 euler_p, rk4_p, newton_p 임
        euler_p = X_static[:, 1]
        rk4_p = X_static[:, 2]
        newton_p = X_static[:, 3]

        # 실제값과의 오차 계산 (작을수록 좋음, 0으로 나누는 것 방지 위해 1e-5 추가)
        err_e = np.abs(y_actual - euler_p) + 1e-5
        err_r = np.abs(y_actual - rk4_p) + 1e-5
        err_n = np.abs(y_actual - newton_p) + 1e-5

        # 오차가 작을수록 가중치가 높아지도록 역수(1/x) 취하기
        inv_e, inv_r, inv_n = 1/err_e, 1/err_r, 1/err_n
        sum_inv = inv_e + inv_r + inv_n

        # 최종 가중치 정답지 (ex: [0.15, 0.65, 0.20])
        y_target_weights = np.vstack([inv_e/sum_inv, inv_r/sum_inv, inv_n/sum_inv]).T
        # ----------------------------------------------------

        # 🎯 [2단계] LSTM에게 시장 상황별 최고의 가중치를 학습시킴
        if self.lstm_model is None:
            self.lstm_model, self.feature_extractor = self.build_attention_lstm((X_seq.shape[1], X_seq.shape[2]))
        
        self.lstm_model.fit(X_seq, y_target_weights, epochs=30, verbose=0, batch_size=32)

        # 🎯 [3단계] 방금 똑똑해진 LSTM으로 '맞춤형 동적 베이스라인' 계산
        pred_weights = self.lstm_model.predict(X_seq, verbose=0)
        dynamic_base = (pred_weights[:, 0] * euler_p) + (pred_weights[:, 1] * rk4_p) + (pred_weights[:, 2] * newton_p)

        # XGBoost가 맞춰야 할 새로운 오차(잔차)
        new_y_residual = y_actual - dynamic_base

        # 🎯 [4단계] 특징 추출 및 XGBoost 학습 (최종 미세 보정)
        lstm_feats = self.feature_extractor.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        
        self.xgb_model.fit(combined_X, new_y_residual)
        self.is_trained = True

    def predict(self, X_seq, X_static):
        if not self.is_trained:
            return None
            
        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_static = np.asarray(X_static, dtype=np.float32)
            
        # 1. AI가 현재 시장에서 가장 신뢰하는 가중치 3개 뱉어냄
        weights = self.lstm_model.predict(X_seq, verbose=0)[0]
        
        # 2. XGBoost가 최종 잔차 보정치 뱉어냄
        lstm_feats = self.feature_extractor.predict(X_seq, verbose=0)
        combined_X = np.hstack([X_static, lstm_feats])
        ai_residual = self.xgb_model.predict(combined_X)[0]
        
        # 결과창 출력을 위해 보정치와 가중치를 함께 반환
        return ai_residual, weights
