import wandb
import pandas as pd
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. GPU 설정 ---
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

# --- 2. 경로 및 전역 설정 ---
CSV_FILE_PATH = "./final_cleaned_interactive.csv" 
NPY_DIR = "./2_npy_timeseries"
PNG_DIR = "./3_audio_spectrograms"
IMG_HEIGHT, IMG_WIDTH = 128, 128
NPY_SEQ_LENGTH, NPY_FEATURES = 90, 5 

# --- 3. Sweep 설정 (여기에 튜닝할 범위를 정의) ---
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
        'batch_size': {'values': [16, 32, 64]},
        'cnn_model': {'values': ['LeNet', 'AlexNet_Mini']}, # VGG는 무거우면 제외 가능
        'cnn_latent_dim': {'values': [64, 128]},
        'rnn_model': {'values': ['LSTM', 'GRU']},
        'rnn_units': {'values': [32, 64]},
        'bottleneck_dim': {'values': [32, 64]}
    }
}

# --- 4. RAM 캐싱 데이터셋 (기존 코드 유지) ---
class MultiModalRamDataset(Dataset):
    def __init__(self, df, npy_dir, png_dir, img_dims, npy_dims, scaler, mode='Train'):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.png_dir = png_dir
        self.img_height, self.img_width = img_dims
        self.seq_len, self.n_features = npy_dims
        self.scaler = scaler
        self.cached_data = []
        
        print(f"[{mode}] 데이터를 RAM에 로드 중... (최초 1회만 실행)")
        for i in tqdm(range(len(self.df)), desc=f"Loading {mode}"):
            video_id = self.df.loc[i, 'video_id']
            
            # PNG 처리
            try:
                png_path = os.path.join(self.png_dir, f"{video_id}.png")
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise FileNotFoundError
                img = cv2.resize(img, (self.img_width, self.img_height))
                img_normalized = img / 255.0
                img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
            except:
                img_tensor = torch.zeros((1, self.img_height, self.img_width), dtype=torch.float32)
            
            # NPY 처리
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
        return self.cached_data[index], self.cached_data[index] # Autoencoder라 입력=정답

# --- 5. 모델 (Sweep 파라미터 적용되도록 수정됨) ---
class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        self.cfg = cfg
        
        # 1) CNN Encoder 선택 (Sweep Config에 따라 변경)
        cnn_type = getattr(cfg, 'cnn_model', 'AlexNet_Mini')
        
        if cnn_type == 'LeNet':
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), # 64x64
                nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), # 32x32
                nn.Flatten(), nn.Linear(32 * 32 * 32, cfg.cnn_latent_dim), nn.ReLU()
            )
        else: # AlexNet_Mini (Default)
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 5, 2, 0), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(64 * 15 * 15, cfg.cnn_latent_dim), nn.ReLU()
            )

        # 2) RNN Encoder 선택
        rnn_type = getattr(cfg, 'rnn_model', 'LSTM')
        if rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size=NPY_FEATURES, hidden_size=cfg.rnn_units, batch_first=True)
        else:
            self.rnn_encoder = nn.LSTM(input_size=NPY_FEATURES, hidden_size=cfg.rnn_units, batch_first=True)
            
        # 3) Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim), nn.ReLU())
        
        # 4) Decoders
        self.rnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, cfg.rnn_units)
        if rnn_type == 'GRU':
            self.rnn_decoder = nn.GRU(input_size=cfg.rnn_units, hidden_size=cfg.rnn_units, batch_first=True)
        else:
            self.rnn_decoder = nn.LSTM(input_size=cfg.rnn_units, hidden_size=cfg.rnn_units, batch_first=True)
        self.rnn_output_layer = nn.Linear(cfg.rnn_units, NPY_FEATURES)
        
        # CNN Decoder (구조 단순화를 위해 공통 사용, 필요시 분기 가능)
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
        
        # RNN 분기 처리 (LSTM은 h, c 반환 / GRU는 h 반환)
        if isinstance(self.rnn_encoder, nn.LSTM):
            _, (h_n, _) = self.rnn_encoder(npy)
        else:
            _, h_n = self.rnn_encoder(npy)
            
        # h_n의 shape: (num_layers, batch, hidden). 마지막 레이어만 사용 -> h_n[-1]
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        
        rnn_in = self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1)
        rnn_out, _ = self.rnn_decoder(rnn_in)
        
        return self.cnn_decoder(self.cnn_decoder_fc(z)), self.rnn_output_layer(rnn_out)

# --- 6. 유틸리티 (EarlyStopping, Scaler) ---
class EarlyStopping:
    def __init__(self, patience=5, delta=0): # Sweep 속도를 위해 patience 5로 조정
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None: self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

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

# --- 7. 데이터 로드 (전역 변수로 선언하여 Sweep 반복 시 재사용) ---
# 전역 변수로 선언해두면 agent가 train_sweep 함수를 여러 번 호출해도 
# 이 부분은 다시 실행되지 않아 RAM 로딩 시간을 아낄 수 있습니다.
print("🚀 [시스템] 데이터셋 초기화 중... (이 과정은 한 번만 수행됩니다)")
if os.path.exists(CSV_FILE_PATH):
    df_all = pd.read_csv(CSV_FILE_PATH)
    df_train, df_val = train_test_split(df_all, test_size=0.2, random_state=42)
    scaler = get_npy_scaler(df_train, NPY_DIR)
    
    # ★ 데이터셋 미리 로드 (RAM 상주)
    train_dataset_global = MultiModalRamDataset(df_train, NPY_DIR, PNG_DIR, (IMG_HEIGHT, IMG_WIDTH), (NPY_SEQ_LENGTH, NPY_FEATURES), scaler, mode='Train')
    val_dataset_global = MultiModalRamDataset(df_val, NPY_DIR, PNG_DIR, (IMG_HEIGHT, IMG_WIDTH), (NPY_SEQ_LENGTH, NPY_FEATURES), scaler, mode='Val')
    print("✅ 데이터 로드 완료!")
else:
    print(f"❌ [오류] CSV 파일이 없습니다: {CSV_FILE_PATH}")
    train_dataset_global, val_dataset_global = None, None

# --- 8. Sweep용 학습 함수 ---
def train_sweep():
    # WandB 초기화 (Sweep Agent가 설정을 주입함)
    wandb.init()
    config = wandb.config
    
    seed_everything(42)
    
    # DataLoader 생성 (Batch Size는 튜닝 대상이므로 매번 새로 생성)
    if train_dataset_global is None: return
    
    train_loader = DataLoader(train_dataset_global, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_global, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 모델 생성
    model = MultiModalAutoencoder(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    early_stopping = EarlyStopping(patience=5)
    
    # 학습 루프 (Epochs는 보통 10~15 정도로 고정하거나 config에 추가 가능)
    epochs = 10 
    
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
        
        # WandB 기록
        wandb.log({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val})
        
        # Early Stopping 체크
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print(f"⏹ Early Stopping at Epoch {epoch+1}")
            break

# --- 9. 메인 실행부 ---
if __name__ == "__main__":
    if train_dataset_global is None:
        print("데이터 로드 실패로 종료합니다.")
        exit()

    print("\n🚀 Sweep Agent 시작! (WandB 대시보드에서 진행상황 확인 가능)")
    
    # Sweep 등록
    sweep_id = wandb.sweep(sweep_config, project="deepfake-LQ-CNN-model_2")
    
    # Agent 실행 (count=10: 총 100번의 다른 조합 시도)
    wandb.agent(sweep_id, function=train_sweep, count=100)