# train_final_torch.py (수정본)

import wandb
import pandas as pd
import numpy as np
import os
import cv2
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- GPU 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 경로 설정 (전역 변수) ---
CSV_FILE_PATH = "./FINAL_master_summary_28828.csv" 
NPY_DIR = "./FINAL_NPY_28828"
PNG_DIR = "./3_audio_spectrograms"
IMG_HEIGHT, IMG_WIDTH = 128, 128
NPY_SEQ_LENGTH, NPY_FEATURES = 90, 5 

# --- Dataset 클래스 (동일) ---
class MultiModalRamDataset(Dataset):
    def __init__(self, df, npy_dir, png_dir, img_dims, npy_dims, scaler, mode='Train'):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.png_dir = png_dir
        self.img_height, self.img_width = img_dims
        self.seq_len, self.n_features = npy_dims
        self.scaler = scaler
        self.cached_data = []
        
        # 데이터 로딩 로그는 너무 많으면 보기 힘드니 간단히 처리
        # print(f"[{mode}] 데이터 로드 중...") 
        
        for i in range(len(self.df)):
            video_id = self.df.loc[i, 'video_id']
            try:
                png_path = os.path.join(self.png_dir, f"{video_id}.png")
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise FileNotFoundError
                img = cv2.resize(img, (self.img_width, self.img_height))
                img_normalized = img / 255.0
                img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
            except:
                img_tensor = torch.zeros((1, self.img_height, self.img_width), dtype=torch.float32)
            
            try:
                npy_path = os.path.join(self.npy_dir, f"{video_id}.npy")
                data = np.load(npy_path, allow_pickle=True).item()
                mouth_data = np.stack([
                    data['mouth']['laplacian_mean'], data['mouth']['laplacian_var'],
                    data['mouth']['light_intensity_mean'], data['mouth']['light_intensity_change'],
                    data['mouth']['area']
                ], axis=1)
                mouth_data_scaled = self.scaler.transform(mouth_data.reshape(-1, self.n_features))
                curr_len = mouth_data_scaled.shape[0]
                padded_data = np.zeros((self.seq_len, self.n_features))
                if curr_len > self.seq_len: padded_data = mouth_data_scaled[:self.seq_len, :]
                else: padded_data[:curr_len, :] = mouth_data_scaled
                npy_tensor = torch.tensor(padded_data, dtype=torch.float32)
            except:
                npy_tensor = torch.zeros((self.seq_len, self.n_features), dtype=torch.float32)
            
            self.cached_data.append((img_tensor, npy_tensor))
            
    def __len__(self): return len(self.cached_data)
    def __getitem__(self, index):
        img_tensor, npy_tensor = self.cached_data[index]
        return (img_tensor, npy_tensor), (img_tensor, npy_tensor)

# --- Model 클래스 (동일) ---
class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        self.cfg = cfg
        
        # CNN 모델 선택 (Sweep 파라미터 대응)
        # config에 cnn_model 값이 문자열로 들어오므로 분기 처리
        if getattr(cfg, 'cnn_model', 'AlexNet_Mini') == 'AlexNet_Mini':
             self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
                nn.Flatten(), nn.Linear(64 * 15 * 15, cfg.cnn_latent_dim), nn.ReLU()
            )
        else: 
            # 다른 모델인 경우 기본 구조 (예시)
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(16 * 64 * 64, cfg.cnn_latent_dim), nn.ReLU()
            )

        # RNN 인코더
        self.rnn_encoder = nn.LSTM(input_size=NPY_FEATURES, hidden_size=cfg.rnn_units, batch_first=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim), nn.ReLU())
        
        # Decoders
        self.rnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, cfg.rnn_units)
        self.rnn_decoder = nn.LSTM(input_size=cfg.rnn_units, hidden_size=cfg.rnn_units, batch_first=True)
        self.rnn_output_layer = nn.Linear(cfg.rnn_units, NPY_FEATURES)
        
        self.cnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, 64 * 16 * 16)
        self.cnn_decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        
    def forward(self, img, npy):
        cnn_feat = self.cnn_encoder(img)
        _, (h_n, _) = self.rnn_encoder(npy)
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        rnn_out, _ = self.rnn_decoder(self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1))
        return self.cnn_decoder(self.cnn_decoder_fc(z)), self.rnn_output_layer(rnn_out)

# --- Scaler 함수 ---
def get_npy_scaler(df, npy_dir):
    scaler_path = "npy_scaler.joblib"
    if os.path.exists(scaler_path):
        from joblib import load
        return load(scaler_path)
    from joblib import dump
    scaler = StandardScaler()
    sample_ids = df['video_id'].sample(min(len(df), 1500), random_state=42)
    all_npy = []
    for vid in sample_ids:
        try:
            d = np.load(os.path.join(npy_dir, f"{vid}.npy"), allow_pickle=True).item()['mouth']
            all_npy.append(np.stack([d['laplacian_mean'], d['laplacian_var'], d['light_intensity_mean'], d['light_intensity_change'], d['area']], axis=1))
        except: pass
    scaler.fit(np.concatenate(all_npy).reshape(-1, NPY_FEATURES))
    dump(scaler, scaler_path)
    return scaler

# --- ★★★ 핵심: Train 함수로 변경 ★★★ ---
def train_sweep():
    # 1. WandB 초기화 (Agent가 설정을 주입해줌)
    wandb.init()
    
    # WandB가 준 설정값(config)을 가져옴
    config = wandb.config
    
    seed_everything(42)
    
    # 2. 데이터 로드 (전역 변수 경로 사용)
    if not os.path.exists(CSV_FILE_PATH):
        print("CSV 없음"); return

    df_all = pd.read_csv(CSV_FILE_PATH)
    df_train, df_val = train_test_split(df_all, test_size=0.2, random_state=42)
    scaler = get_npy_scaler(df_train, NPY_DIR)
    
    # 데이터셋 & 로더 (Batch Size는 Sweep config에서 가져옴)
    train_dataset = MultiModalRamDataset(df_train, NPY_DIR, PNG_DIR, (IMG_HEIGHT, IMG_WIDTH), (NPY_SEQ_LENGTH, NPY_FEATURES), scaler, mode='Train')
    val_dataset = MultiModalRamDataset(df_val, NPY_DIR, PNG_DIR, (IMG_HEIGHT, IMG_WIDTH), (NPY_SEQ_LENGTH, NPY_FEATURES), scaler, mode='Val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 3. 모델 생성
    model = MultiModalAutoencoder(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 4. 학습 루프 (Epochs도 config에 있다면 config.epochs, 없으면 고정값)
    epochs = getattr(config, 'epochs', 15) # Sweep 테스트용으로 15회 정도로 줄임 (권장)
    
    print(f"🚀 Sweep Start: LR={config.learning_rate}, BS={config.batch_size}, Model={config.cnn_model}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for (img_in, npy_in), (img_t, npy_t) in train_loader:
            img_in, npy_in, img_t, npy_t = img_in.to(device), npy_in.to(device), img_t.to(device), npy_t.to(device)
            optimizer.zero_grad()
            p_out, n_out = model(img_in, npy_in)
            loss = criterion(p_out, img_t) + criterion(n_out, npy_t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (img_in, npy_in), (img_t, npy_t) in val_loader:
                img_in, npy_in, img_t, npy_t = img_in.to(device), npy_in.to(device), img_t.to(device), npy_t.to(device)
                p_out, n_out = model(img_in, npy_in)
                val_loss += (criterion(p_out, img_t) + criterion(n_out, npy_t)).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        # WandB에 기록 (매 Epoch 마다)
        wandb.log({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val})
        
    print(f"✨ Sweep Run Finished: Val Loss = {avg_val:.6f}")

# if __name__ == "__main__": 부분은 이제 필요 없습니다.
# 별도의 실행 파일에서 호출할 것입니다.