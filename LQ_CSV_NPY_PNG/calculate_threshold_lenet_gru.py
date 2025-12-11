# calculate_threshold_lenet_gru.py
# (LeNet + GRU 모델용 임계값 계산 코드)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
from joblib import load
from tqdm import tqdm

# --- 1. 설정 및 경로 (train_best_model.py와 동일하게 설정) ---
config = SimpleNamespace(
    batch_size = 256,          # 추론 시에는 배치를 키워도 무방함
    bottleneck_dim = 64,
    cnn_latent_dim = 128,      # 학습 코드에 맞춰 수정 (64 -> 128)
    cnn_model = "LeNet",       # 학습 코드에 맞춰 수정
    rnn_model = "GRU",         # 학습 코드에 맞춰 수정
    rnn_units = 64
)

# 파일 경로 (학습 코드 기준)
CSV_FILE_PATH = "./final_cleaned_interactive.csv" 
NPY_DIR = "./2_npy_timeseries"
PNG_DIR = "./3_audio_spectrograms"
MODEL_PATH = "best_deepfake_model.pt"      # 학습된 모델 파일
SCALER_PATH = "npy_scaler_final.joblib"    # 학습된 스케일러 파일

IMG_HEIGHT, IMG_WIDTH = 128, 128
NPY_SEQ_LENGTH, NPY_FEATURES = 90, 5 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 모델 클래스 정의 (train_best_model.py와 100% 일치) ---
class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        
        # 1) CNN Encoder: LeNet
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), # 64x64
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), # 32x32
            nn.Flatten(), 
            nn.Linear(32 * 32 * 32, cfg.cnn_latent_dim), nn.ReLU()
        )

        # 2) RNN Encoder: GRU
        self.rnn_encoder = nn.GRU(input_size=NPY_FEATURES, hidden_size=cfg.rnn_units, batch_first=True)
            
        # 3) Bottleneck (Fusion)
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim), 
            nn.ReLU()
        )
        
        # 4) RNN Decoder (GRU)
        self.rnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, cfg.rnn_units)
        self.rnn_decoder = nn.GRU(input_size=cfg.rnn_units, hidden_size=cfg.rnn_units, batch_first=True)
        self.rnn_output_layer = nn.Linear(cfg.rnn_units, NPY_FEATURES)
        
        # 5) CNN Decoder
        self.cnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, 64 * 16 * 16)
        self.cnn_decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        
    def forward(self, img, npy):
        # Encoding
        cnn_feat = self.cnn_encoder(img)
        _, h_n = self.rnn_encoder(npy) # GRU는 h_n만 반환
        
        # Fusion
        # h_n shape: (num_layers, batch, hidden) -> 맨 마지막 레이어 사용
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        
        # Decoding (RNN)
        rnn_in = self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1)
        rnn_out, _ = self.rnn_decoder(rnn_in)
        
        # Decoding (CNN)
        cnn_out = self.cnn_decoder(self.cnn_decoder_fc(z))
        
        return cnn_out, self.rnn_output_layer(rnn_out)

# --- 3. 데이터셋 정의 (추론용) ---
class InferenceDataset(Dataset):
    def __init__(self, df, npy_dir, png_dir, scaler):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.png_dir = png_dir
        self.scaler = scaler
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        video_id = self.df.loc[index, 'video_id']
        
        # PNG 로드 (학습 코드와 동일 전처리)
        try:
            path = os.path.join(self.png_dir, f"{video_id}.png")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: raise Exception
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0  # Normalize
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        except:
            img_tensor = torch.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=torch.float32)

        # NPY 로드 (학습 코드와 동일 전처리)
        try:
            path = os.path.join(self.npy_dir, f"{video_id}.npy")
            d = np.load(path, allow_pickle=True).item()['mouth']
            mouth = np.stack([
                d['laplacian_mean'], d['laplacian_var'], 
                d['light_intensity_mean'], d['light_intensity_change'], 
                d['area']
            ], axis=1)
            
            # 스케일러 적용
            mouth_s = self.scaler.transform(mouth.reshape(-1, NPY_FEATURES))
            
            # 패딩/자르기
            pad = np.zeros((NPY_SEQ_LENGTH, NPY_FEATURES))
            length = min(len(mouth_s), NPY_SEQ_LENGTH)
            pad[:length, :] = mouth_s[:length, :]
            npy_tensor = torch.tensor(pad, dtype=torch.float32)
        except:
            npy_tensor = torch.zeros((NPY_SEQ_LENGTH, NPY_FEATURES), dtype=torch.float32)
            
        return img_tensor, npy_tensor, video_id

# --- 4. 메인 실행: 임계값 계산 ---
if __name__ == "__main__":
    print(f"🚀 [임계값 계산] 시작 (Device: {device})")
    print(f"   - Model: LeNet + GRU")
    print(f"   - Weights: {MODEL_PATH}")
    
    # 1. 데이터 준비 (Validation Set만 사용)
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {CSV_FILE_PATH}")
        exit()

    df_all = pd.read_csv(CSV_FILE_PATH)
    _, df_val = train_test_split(df_all, test_size=0.2, random_state=42)
    print(f"📊 검증 데이터 개수: {len(df_val)}개")
    
    # 스케일러 로드
    if not os.path.exists(SCALER_PATH):
        print(f"❌ 스케일러 파일을 찾을 수 없습니다: {SCALER_PATH}")
        exit()
    scaler = load(SCALER_PATH)
    
    val_dataset = InferenceDataset(df_val, NPY_DIR, PNG_DIR, scaler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 2. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        exit()
        
    model = MultiModalAutoencoder(config).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ 학습된 모델 로드 완료")
    
    # 3. 복원 오차(Loss) 계산
    losses = []
    video_ids = []
    # 개별 샘플의 Loss를 구하기 위해 reduction='none' 사용하지 않고 직접 계산하거나 
    # reduction='none' 후 mean을 취함. 여기선 직접 계산.
    
    print("🔍 복원 오차 계산 중...")
    with torch.no_grad():
        for img, npy, v_ids in tqdm(val_loader):
            img, npy = img.to(device), npy.to(device)
            
            # Forward
            cnn_out, rnn_out = model(img, npy)
            
            # Loss 계산 (Batch 내 각 샘플별 MSE)
            # 이미지 Loss: (B, 1, H, W) -> (B,)
            loss_p = torch.mean((cnn_out - img)**2, dim=[1, 2, 3]) 
            # NPY Loss: (B, Seq, Feat) -> (B,)
            loss_n = torch.mean((rnn_out - npy)**2, dim=[1, 2])
            
            # Total Loss (두 오차의 합)
            total_loss = loss_p + loss_n
            
            losses.extend(total_loss.cpu().numpy())
            video_ids.extend(v_ids)
            
    # 4. 통계 분석 및 시각화
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    max_loss = np.max(losses)
    
    print("\n" + "="*40)
    print(f"📊 [정상 데이터 복원 오차 통계 - LeNet+GRU]")
    print(f" - 평균(Mean): {mean_loss:.6f}")
    print(f" - 표준편차(Std): {std_loss:.6f}")
    print(f" - 최소(Min): {np.min(losses):.6f}")
    print(f" - 최대(Max): {max_loss:.6f}")
    print("="*40)
    
    # 5. 추천 임계값 제안
    threshold_2std = mean_loss + 2 * std_loss
    threshold_3std = mean_loss + 3 * std_loss
    threshold_max = max_loss
    
    print(f"\n💡 [추천 임계값(Threshold)]")
    print(f"1️⃣ 느슨한 기준 (Mean + 2σ): {threshold_2std:.6f} (민감하게 탐지)")
    print(f"2️⃣ 엄격한 기준 (Mean + 3σ): {threshold_3std:.6f} (확실한 이상치만 탐지)")
    print(f"3️⃣ 최대값 기준 (Max Val):    {threshold_max:.6f} (가장 보수적, 오탐지 최소화)")
    
    # 6. 히스토그램 그리기
    plt.figure(figsize=(10, 6))
    sns.histplot(losses, bins=50, kde=True, color='green', label='Normal Data (Val)')
    plt.axvline(threshold_2std, color='orange', linestyle='--', label=f'Threshold (2std): {threshold_2std:.4f}')
    plt.axvline(threshold_3std, color='red', linestyle='--', label=f'Threshold (3std): {threshold_3std:.4f}')
    plt.title("Reconstruction Error Distribution (LeNet + GRU)")
    plt.xlabel("Reconstruction Loss (MSE)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "threshold_distribution_lenet_gru.png"
    plt.savefig(save_path)
    print(f"\n📈 히스토그램 저장 완료: {save_path}")
    print("결과 그래프를 확인하고 시스템에 적용할 임계값을 선택하세요.")