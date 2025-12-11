# train_multimodal_ae.py
# (비정상 탐지를 위한 멀티모달 오토인코더 학습 스크립트)

import wandb
import pandas as pd
import numpy as np
import os
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# TensorFlow 및 Keras 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv2D, MaxPooling2D, Flatten, 
    Dropout, BatchNormalization, concatenate, Reshape,
    Conv2DTranspose, RepeatVector, TimeDistributed
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger # Wandb 2.0+ 최신 콜백

# --- 1. WandB Sweep 설정 불러오기 ---
from sweep_config_ae import sweep_config 

# --- 2. 고정 경로 및 설정 ---
# (이전 스크립트 v9의 출력 경로와 일치해야 함)

# ⚠️ [경로 확인 1] 최종 CSV 파일 (28,828개 ID)
CSV_FILE_PATH = "./FINAL_master_summary_28828.csv" 

# ⚠️ [경로 확인 2] 최종 NPY 폴더 (28,828개 파일)
NPY_DIR = "./FINAL_NPY_28828"

# ⚠️ [경로 확인 3] 최종 PNG 폴더 (28,828개 파일)
PNG_DIR = "./3_audio_spectrograms"

# 모델 입력 크기 설정
IMG_HEIGHT = 128
IMG_WIDTH = 128  # PNG 스펙트로그램의 크기
NPY_SEQ_LENGTH = 90 # NPY 파일의 프레임 수 (가정)
NPY_FEATURES = 5    # NPY에서 'mouth' 관련 5개 특징 (lap_mean, lap_var, light_mean, light_change, area)

# --- 3. 커스텀 데이터 생성기 (Autoencoder용) ---
# [수정됨] 딕셔너리 반환 및 패딩/절삭 기능 추가
class MultiModalDataGenerator(Sequence):
    def __init__(self, df, npy_dir, png_dir, batch_size,
                 img_dims, npy_dims, scaler, is_train=True):
        self.df = df.copy()
        self.npy_dir = npy_dir
        self.png_dir = png_dir
        self.batch_size = batch_size
        self.img_height, self.img_width = img_dims
        self.seq_len, self.n_features = npy_dims
        self.scaler = scaler 
        self.is_train = is_train
        
        self.ids = self.df['video_id'].values
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # 'index'번째 배치를 생성
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_ids = self.ids[start_idx:end_idx]
        
        X_img = np.empty((self.batch_size, self.img_height, self.img_width, 1))
        X_npy = np.empty((self.batch_size, self.seq_len, self.n_features))
        
        for i, video_id in enumerate(batch_ids):
            try:
                # 1. PNG 로드 및 전처리
                png_path = os.path.join(self.png_dir, f"{video_id}.png")
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.img_width, self.img_height))
                img_normalized = img / 255.0 
                X_img[i,] = np.expand_dims(img_normalized, axis=-1)
                
                # 2. NPY 로드 및 전처리
                npy_path = os.path.join(self.npy_dir, f"{video_id}.npy")
                data = np.load(npy_path, allow_pickle=True).item()
                
                mouth_data = np.stack([
                    data['mouth']['laplacian_mean'],
                    data['mouth']['laplacian_var'],
                    data['mouth']['light_intensity_mean'],
                    data['mouth']['light_intensity_change'],
                    data['mouth']['area']
                ], axis=1) 
                
                # [수정됨] 1. 원본 데이터 스케일링 (크기가 어떻든)
                mouth_data_scaled = self.scaler.transform(mouth_data.reshape(-1, self.n_features))
                mouth_data_scaled = mouth_data_scaled.reshape(-1, self.n_features) # (N, 5)

                # [수정됨] 2. 스케일링된 데이터를 (90, 5)로 패딩/절삭
                current_len = mouth_data_scaled.shape[0] # (예: 89 또는 360)
                target_len = self.seq_len                # (90)
                
                # (90, 5) 크기의 0으로 채워진 배열 생성
                padded_data_scaled = np.zeros((target_len, self.n_features))
                
                if current_len > target_len: # 절삭 (예: 360 -> 90)
                    padded_data_scaled = mouth_data_scaled[:target_len, :]
                else: # 패딩 (예: 89 -> 90)
                    # 89 프레임만 복사 (마지막 1 프레임은 0으로 남음)
                    padded_data_scaled[:current_len, :] = mouth_data_scaled

                X_npy[i,] = padded_data_scaled
                
            except Exception as e:
                print(f"\n[데이터 로드 오류] ID: {video_id} / 오류: {e}")
                X_img[i,] = 0
                X_npy[i,] = 0

        # [수정됨] X(입력)를 리스트가 아닌 딕셔너리로 반환
        # (Y(출력)는 원래 딕셔너리였음)
        X_inputs = {'png_input': X_img, 'npy_input': X_npy}
        Y_outputs = {'png_output': X_img, 'npy_output': X_npy}
        
        return X_inputs, Y_outputs

    def on_epoch_end(self):
        if self.is_train:
            np.random.shuffle(self.ids)

# --- 4. NPY 데이터 스케일러 준비 ---
def get_npy_scaler(df, npy_dir):
    """'학습 데이터'의 NPY 통계치(mean, std)를 계산 (단 1회 실행)"""
    print("\n[전처리] NPY 데이터 스케일러(StandardScaler) 피팅 시작...")
    
    scaler_path = "npy_scaler.joblib"
    
    # (시간 절약) 이미 스케일러 파일이 있으면 로드
    if os.path.exists(scaler_path):
        from joblib import load
        scaler = load(scaler_path)
        print("✓ 저장된 NPY 스케일러(npy_scaler.joblib)를 로드했습니다.")
        return scaler
        
    print("(학습 데이터의 NPY 파일을 읽어 통계치를 계산합니다...)")
    from joblib import dump
    scaler = StandardScaler()
    sample_ids = df['video_id'].sample(min(len(df), 1500), random_state=42)
    all_npy_data = []
    
    for video_id in tqdm(sample_ids, desc="NPY 샘플 읽는 중"):
        try:
            npy_path = os.path.join(NPY_DIR, f"{video_id}.npy")
            data = np.load(npy_path, allow_pickle=True).item()
            
            mouth_data = np.stack([
                data['mouth']['laplacian_mean'],
                data['mouth']['laplacian_var'],
                data['mouth']['light_intensity_mean'],
                data['mouth']['light_intensity_change'],
                data['mouth']['area']
            ], axis=1)
            all_npy_data.append(mouth_data)
        except Exception:
            pass
            
    combined_data = np.concatenate(all_npy_data).reshape(-1, NPY_FEATURES)
    scaler.fit(combined_data)
    
    # 스케일러를 파일로 저장
    dump(scaler, scaler_path) 
    print(f"✓ NPY 스케일러 피팅 완료 및 '{scaler_path}'에 저장.")
    return scaler

# --- 5. 모델 빌더 (인코더/디코더) ---

# 5-1. CNN 인코더 브랜치
def build_cnn_encoder(config, img_input):
    model_type = config.cnn_model
    latent_dim = config.cnn_latent_dim
    
    if model_type == 'LeNet':
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(latent_dim, activation='relu')(x)
        return x

    elif model_type == 'AlexNet_Mini':
        x = Conv2D(32, (5, 5), strides=(2,2), activation='relu')(img_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(latent_dim, activation='relu')(x)
        return x

    elif model_type == 'VGG_Mini':
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) # 64x64
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) # 32x32
        x = Flatten()(x)
        x = Dense(latent_dim, activation='relu')(x)
        return x

# 5-2. RNN 인코더 브랜치
def build_rnn_encoder(config, npy_input):
    model_type = config.rnn_model
    units = config.rnn_units
    
    if model_type == 'LSTM':
        x = LSTM(units, return_sequences=False)(npy_input) 
        return x
    elif model_type == 'GRU':
        x = GRU(units, return_sequences=False)(npy_input)
        return x

# 5-3. 디코더 (복원)
def build_decoders(bottleneck_vector, config):
    
    # --- RNN 디코더 (NPY 복원) ---
    # (Bottleneck -> 90, 5)
    rnn_units = config.rnn_units
    
    # 병목 벡터를 RNN 유닛 수만큼 확장
    x_rnn = Dense(rnn_units, activation='relu')(bottleneck_vector)
    # (90, rnn_units) 형태로 NPY 시퀀스 길이만큼 복제
    x_rnn = RepeatVector(NPY_SEQ_LENGTH)(x_rnn) 
    
    if config.rnn_model == 'LSTM':
        x_rnn = LSTM(rnn_units, return_sequences=True)(x_rnn)
    else:
        x_rnn = GRU(rnn_units, return_sequences=True)(x_rnn)
        
    # (90, 5) 형태로 복원 (StandardScaler로 스케일링된 값)
    npy_decoded = TimeDistributed(Dense(NPY_FEATURES, activation='linear'), name='npy_output')(x_rnn)
    
    
    # --- CNN 디코더 (PNG 복원) ---
    # (Bottleneck -> 128, 128, 1)
    # 디코더가 시작할 적절한 3D 형태(예: 16x16x64)로 Dense 확장
    start_shape = (16, 16, 64)
    x_cnn = Dense(start_shape[0] * start_shape[1] * start_shape[2], activation='relu')(bottleneck_vector)
    x_cnn = Reshape(start_shape)(x_cnn)
    
    # 16x16 -> 32x32
    x_cnn = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x_cnn)
    # 32x32 -> 64x64
    x_cnn = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x_cnn)
    # 64x64 -> 128x128
    x_cnn = Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x_cnn)
    
    # (128, 128, 1) 형태로 복원 (0~1 사이의 값)
    png_decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='png_output')(x_cnn)
    
    return png_decoded, npy_decoded

# 5-4. 오토인코더 모델 결합
def build_autoencoder(config):
    
    # --- 입력 ---
    img_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='png_input')
    npy_input = Input(shape=(NPY_SEQ_LENGTH, NPY_FEATURES), name='npy_input')
    
    # --- 1. 인코더 ---
    cnn_encoded = build_cnn_encoder(config, img_input)
    rnn_encoded = build_rnn_encoder(config, npy_input)
    
    # --- 2. 결합 및 병목 ---
    combined = concatenate([cnn_encoded, rnn_encoded])
    bottleneck = Dense(config.bottleneck_dim, activation='relu', name='bottleneck')(combined)
    
    # --- 3. 디코더 ---
    png_decoded, npy_decoded = build_decoders(bottleneck, config)
    
    # --- 모델 생성 ---
    model = Model(inputs=[img_input, npy_input], outputs=[png_decoded, npy_decoded])
    
    # --- 컴파일 ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    model.compile(
        loss={
            'png_output': 'mse',  # PNG 복원 오류
            'npy_output': 'mse'   # NPY 복원 오류
        },
        loss_weights={
            'png_output': 1.0,  # PNG 오류 가중치
            'npy_output': 1.0   # NPY 오류 가중치 (두 손실을 1:1로 반영)
        },
        optimizer=optimizer
    )
    
    return model

# --- 6. WandB Sweep을 위한 메인 학습 함수 ---

def train():
    """WandB Agent가 호출할 메인 학습 함수"""
    
    # 1. WandB 초기화 (Sweep에서 하이퍼파라미터 받아옴)
    wandb.init()
    config = wandb.config # 현재 Sweep의 하이퍼파라미터

    print("\n" + "="*50)
    print(f"Sweep 시작: {wandb.run.name}")
    print("현재 하이퍼파라미터:")
    print(config)
    print("="*50)
    
    # 2. 데이터 준비
    try:
        df_all = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 3. 학습 / 검증 데이터 분리 (8:2)
    df_train, df_val = train_test_split(
        df_all, 
        test_size=0.2, # 20%를 검증용으로
        random_state=42
    )
    
    # 4. NPY 스케일러 준비 (학습 데이터(df_train)로만 피팅)
    scaler = get_npy_scaler(df_train, NPY_DIR)
    
    # 5. 데이터 생성기(Generator) 준비
    train_gen = MultiModalDataGenerator(
        df_train, NPY_DIR, PNG_DIR,
        batch_size=config.batch_size,
        img_dims=(IMG_HEIGHT, IMG_WIDTH),
        npy_dims=(NPY_SEQ_LENGTH, NPY_FEATURES),
        scaler=scaler,
        is_train=True
    )
    val_gen = MultiModalDataGenerator(
        df_val, NPY_DIR, PNG_DIR,
        batch_size=config.batch_size,
        img_dims=(IMG_HEIGHT, IMG_WIDTH),
        npy_dims=(NPY_SEQ_LENGTH, NPY_FEATURES),
        scaler=scaler,
        is_train=False
    )
    
    # 6. 모델 구축
    model = build_autoencoder(config)
    model.summary() # 모델 구조 출력

    # 7. 콜백 설정 (조기 종료 및 WandB 로깅)
    callbacks = [
        # 5 에포크 동안 val_loss(총 복원 오류)가 개선되지 않으면 조기 종료
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        WandbMetricsLogger(log_freq='epoch') 
    ]
    
    # 8. 모델 학습
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30, 
        callbacks=callbacks
        # workers 및 use_multiprocessing 인수는 최신 Keras에서 제거됨
    )
    print(f"Sweep 종료: {wandb.run.name}")
    wandb.finish()


# --- 7. 메인 실행 블록 ---
if __name__ == "__main__":
    
    # [필수] 1. WandB 로그인 (터미널에서 'wandb login'을 미리 실행해도 됨)
    # wandb.login() 
    
    print("--- 멀티모달(AE) 딥페이크 탐지 모델 (비정상 탐지) ---")
    print("--- WandB 하이퍼파라미터 튜닝(Sweep) 시작 ---")
    
    # [필수] 2. CSV, NPY, PNG 경로가 올바른지 확인
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 치명적 오류: '{CSV_FILE_PATH}'를 찾을 수 없습니다.")
    elif not os.path.exists(NPY_DIR) or not os.path.exists(PNG_DIR):
        print(f"❌ 치명적 오류: NPY 또는 PNG 폴더를 찾을 수 없습니다.")
    else:
        # 3. WandB Sweep 생성
        sweep_id = wandb.sweep(sweep_config, project="deepfake-multimodal-AE")

        print(f"\n✓ WandB Sweep 생성 완료 (ID: {sweep_id})")
        
        # 4. WandB 에이전트 실행 (count=10: 총 10가지 조합을 테스트)
        print("WandB 에이전트를 시작합니다 (총 10회 실행)...")
        wandb.agent(sweep_id, function=train, count=10)

        print("\n🎉 모든 Sweep 실행이 완료되었습니다.")