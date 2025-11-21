# calculate_threshold.py
# (학습된 모델을 불러와서 정상 데이터의 복원 오차 분포 확인 및 임계값 설정)

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

# --- 1. 설정 및 경로 (학습 때와 동일해야 함) ---
config = SimpleNamespace(
    batch_size = 256, # 추론 때는 커도 상관없음
    bottleneck_dim = 64,
    cnn_latent_dim = 64,
    cnn_model = "AlexNet_Mini",
    rnn_model = "LSTM",
    rnn_units = 64
)

# 파일 경로 확인
CSV_FILE_PATH = "./FINAL_master_summary_28828.csv" 
NPY_DIR = "./FINAL_NPY_28828"
PNG_DIR = "./3_audio_spectrograms"
MODEL_PATH = "best_multimodal_ae_torch_ram.pt" # 저장된 모델 파일명 확인!
SCALER_PATH = "npy_scaler.joblib"

IMG_HEIGHT, IMG_WIDTH = 128, 128
NPY_SEQ_LENGTH, NPY_FEATURES = 90, 5 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 모델 클래스 정의 (학습 코드와 구조가 100% 일치해야 함) ---
class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        self.cfg = cfg
        
        # CNN Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(), nn.Linear(64 * 15 * 15, cfg.cnn_latent_dim), nn.ReLU()
        )
        # RNN Encoder
        self.rnn_encoder = nn.LSTM(input_size=NPY_FEATURES, hidden_size=cfg.rnn_units, batch_first=True)
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim), nn.ReLU())
        
        # RNN Decoder
        self.rnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, cfg.rnn_units)
        self.rnn_decoder = nn.LSTM(input_size=cfg.rnn_units, hidden_size=cfg.rnn_units, batch_first=True)
        self.rnn_output_layer = nn.Linear(cfg.rnn_units, NPY_FEATURES)
        
        # CNN Decoder
        self.cnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, 64 * 16 * 16)
        self.cnn_decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, img, npy):
        cnn_feat = self.cnn_encoder(img)
        _, (h_n, _) = self.rnn_encoder(npy)
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        
        rnn_out, _ = self.rnn_decoder(self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1))
        return self.cnn_decoder(self.cnn_decoder_fc(z)), self.rnn_output_layer(rnn_out)

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
        
        # PNG
        try:
            path = os.path.join(self.png_dir, f"{video_id}.png")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: raise Exception
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        except:
            img_tensor = torch.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=torch.float32)

        # NPY
        try:
            path = os.path.join(self.npy_dir, f"{video_id}.npy")
            d = np.load(path, allow_pickle=True).item()['mouth']
            mouth = np.stack([d['laplacian_mean'], d['laplacian_var'], d['light_intensity_mean'], d['light_intensity_change'], d['area']], axis=1)
            mouth_s = self.scaler.transform(mouth.reshape(-1, NPY_FEATURES))
            
            pad = np.zeros((NPY_SEQ_LENGTH, NPY_FEATURES))
            length = min(len(mouth_s), NPY_SEQ_LENGTH)
            pad[:length, :] = mouth_s[:length, :]
            npy_tensor = torch.tensor(pad, dtype=torch.float32)
        except:
            npy_tensor = torch.zeros((NPY_SEQ_LENGTH, NPY_FEATURES), dtype=torch.float32)
            
        return img_tensor, npy_tensor, video_id

# --- 4. 메인 실행: 임계값 계산 ---
if __name__ == "__main__":
    print(f"🚀 임계값 계산 시작 (Device: {device})")
    
    # 1. 데이터 준비 (Validation Set만 사용)
    df_all = pd.read_csv(CSV_FILE_PATH)
    _, df_val = train_test_split(df_all, test_size=0.2, random_state=42)
    print(f"📊 검증 데이터 개수: {len(df_val)}개")
    
    scaler = load(SCALER_PATH)
    val_dataset = InferenceDataset(df_val, NPY_DIR, PNG_DIR, scaler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 2. 모델 로드
    model = MultiModalAutoencoder(config).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ 학습된 모델 로드 완료")
    
    # 3. 복원 오차(Loss) 계산
    losses = []
    video_ids = []
    criterion = nn.MSELoss(reduction='none') # 샘플별 Loss 계산을 위해 reduction='none'
    
    print("🔍 복원 오차 계산 중...")
    with torch.no_grad():
        for img, npy, v_ids in tqdm(val_loader):
            img, npy = img.to(device), npy.to(device)
            
            # Forward
            p_out, n_out = model(img, npy)
            
            # Loss 계산 (평균이 아니라 개별 샘플의 Loss 합)
            # 이미지 Loss: (B, 1, 128, 128) -> (B,)
            loss_p = torch.mean((p_out - img)**2, dim=[1, 2, 3]) 
            # NPY Loss: (B, 90, 5) -> (B,)
            loss_n = torch.mean((n_out - npy)**2, dim=[1, 2])
            
            # Total Loss (단순 합 또는 가중치 적용)
            total_loss = loss_p + loss_n
            
            losses.extend(total_loss.cpu().numpy())
            video_ids.extend(v_ids)
            
    # 4. 통계 분석 및 시각화
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    max_loss = np.max(losses)
    
    print("\n" + "="*30)
    print(f"📊 [정상 데이터 복원 오차 통계]")
    print(f" - 평균(Mean): {mean_loss:.4f}")
    print(f" - 표준편차(Std): {std_loss:.4f}")
    print(f" - 최소(Min): {np.min(losses):.4f}")
    print(f" - 최대(Max): {max_loss:.4f}")
    print("="*30)
    
    # 5. 추천 임계값 제안
    # 방법 1: 평균 + 2 * 표준편차 (약 95% 커버)
    threshold_2std = mean_loss + 2 * std_loss
    # 방법 2: 평균 + 3 * 표준편차 (약 99% 커버, 보수적)
    threshold_3std = mean_loss + 3 * std_loss
    # 방법 3: 최대값 (데이터가 깨끗하다면 가장 안전)
    threshold_max = max_loss
    
    print(f"\n💡 [추천 임계값(Threshold)]")
    print(f"1️⃣ 느슨한 기준 (Mean + 2σ): {threshold_2std:.4f} (이 값 이상이면 의심)")
    print(f"2️⃣ 엄격한 기준 (Mean + 3σ): {threshold_3std:.4f} (확실한 이상치만 탐지)")
    print(f"3️⃣ 최대값 기준 (Max Val):    {threshold_max:.4f} (Validation 데이터 내 모든 정상 케이스 포함)")
    
    # 6. 히스토그램 그리기
    plt.figure(figsize=(10, 6))
    sns.histplot(losses, bins=50, kde=True, color='blue', label='Normal Data')
    plt.axvline(threshold_2std, color='orange', linestyle='--', label=f'Threshold (2std): {threshold_2std:.2f}')
    plt.axvline(threshold_3std, color='red', linestyle='--', label=f'Threshold (3std): {threshold_3std:.2f}')
    plt.title("Reconstruction Error Distribution (Normal Data)")
    plt.xlabel("Reconstruction Loss (MSE)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "threshold_distribution.png"
    plt.savefig(save_path)
    print(f"\n📈 히스토그램 저장 완료: {save_path}")
    print("이 그래프를 보고 적절한 임계값을 선택하세요.")