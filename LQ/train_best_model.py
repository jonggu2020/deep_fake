import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from types import SimpleNamespace

# --- 1. 최적의 하이퍼파라미터 설정 (사용자 발굴) ---
config = SimpleNamespace(
    batch_size = 32,
    bottleneck_dim = 64,
    cnn_latent_dim = 128,
    cnn_model = "LeNet",       # 가벼운 모델 선택
    learning_rate = 0.0009592063990599656,
    rnn_model = "GRU",         # GRU 선택
    rnn_units = 64,
    epochs = 100               # 충분히 학습하도록 늘림
)

# --- 2. 경로 설정 (환경에 맞게 확인 필요) ---
CSV_FILE_PATH = "./final_cleaned_interactive.csv" 
NPY_DIR = "./2_npy_timeseries"
PNG_DIR = "./3_audio_spectrograms"
IMG_HEIGHT, IMG_WIDTH = 128, 128
NPY_SEQ_LENGTH, NPY_FEATURES = 90, 5 

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

seed_everything(42)

# --- 3. 데이터셋 클래스 (RAM 최적화) ---
class MultiModalRamDataset(Dataset):
    def __init__(self, df, npy_dir, png_dir, img_dims, npy_dims, scaler, mode='Train'):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.png_dir = png_dir
        self.img_height, self.img_width = img_dims
        self.seq_len, self.n_features = npy_dims
        self.scaler = scaler
        self.cached_data = []
        
        print(f"[{mode}] 데이터를 메모리에 올리는 중... (잠시만 기다려주세요)")
        for i in tqdm(range(len(self.df)), desc=f"Loading {mode}"):
            video_id = self.df.loc[i, 'video_id']
            
            # 1. PNG (이미지) 로드
            try:
                png_path = os.path.join(self.png_dir, f"{video_id}.png")
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise FileNotFoundError
                img = cv2.resize(img, (self.img_width, self.img_height))
                img_normalized = img / 255.0
                img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
            except:
                img_tensor = torch.zeros((1, self.img_height, self.img_width), dtype=torch.float32)
            
            # 2. NPY (시계열) 로드
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
        # Autoencoder는 입력이 곧 정답(Target)입니다.
        data = self.cached_data[index]
        return data, data 

# --- 4. 모델 정의 (LeNet + GRU 적용) ---
class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        
        # 1) CNN Encoder: LeNet
        # (사용자가 선택한 LeNet 구조 적용)
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

# --- 5. Early Stopping & Scaler ---
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, path='best_deepfake_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'   ⚠️ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'   ✅ Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_npy_scaler(df, npy_dir):
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    
    scaler_path = "npy_scaler_final.joblib"
    if os.path.exists(scaler_path):
        return load(scaler_path)
        
    print("📊 Scaler 학습 중...")
    scaler = StandardScaler()
    sample_ids = df['video_id'].sample(min(len(df), 2000), random_state=42)
    all_npy = []
    for vid in sample_ids:
        try:
            d = np.load(os.path.join(npy_dir, f"{vid}.npy"), allow_pickle=True).item()['mouth']
            all_npy.append(np.stack([d['laplacian_mean'], d['laplacian_var'], d['light_intensity_mean'], d['light_intensity_change'], d['area']], axis=1))
        except: pass
    scaler.fit(np.concatenate(all_npy).reshape(-1, NPY_FEATURES))
    dump(scaler, scaler_path)
    return scaler

# --- 6. 메인 실행 ---
if __name__ == "__main__":
    print("🚀 [최종 학습] 최적의 파라미터로 모델 학습을 시작합니다!")
    print(f"   - Model: LeNet + GRU")
    print(f"   - Epochs: {config.epochs}")
    print(f"   - Batch Size: {config.batch_size}")
    
    # 1. 데이터 로드
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ CSV 파일 없음: {CSV_FILE_PATH}"); exit()
        
    df_all = pd.read_csv(CSV_FILE_PATH)
    df_train, df_val = train_test_split(df_all, test_size=0.2, random_state=42)
    
    scaler = get_npy_scaler(df_train, NPY_DIR)
    
    train_dataset = MultiModalRamDataset(df_train, NPY_DIR, PNG_DIR, (IMG_HEIGHT, IMG_WIDTH), (NPY_SEQ_LENGTH, NPY_FEATURES), scaler, mode='Train')
    val_dataset = MultiModalRamDataset(df_val, NPY_DIR, PNG_DIR, (IMG_HEIGHT, IMG_WIDTH), (NPY_SEQ_LENGTH, NPY_FEATURES), scaler, mode='Val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 2. 모델 생성
    model = MultiModalAutoencoder(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 3. 학습 (Early Stopping 적용)
    early_stopping = EarlyStopping(patience=15, verbose=True, path='best_deepfake_model.pt')
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for (img_in, npy_in), (img_t, npy_t) in loop:
            img_in, npy_in, img_t, npy_t = img_in.to(device), npy_in.to(device), img_t.to(device), npy_t.to(device)
            
            optimizer.zero_grad()
            cnn_out, rnn_out = model(img_in, npy_in)
            loss = criterion(cnn_out, img_t) + criterion(rnn_out, npy_t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (img_in, npy_in), (img_t, npy_t) in val_loader:
                img_in, npy_in, img_t, npy_t = img_in.to(device), npy_in.to(device), img_t.to(device), npy_t.to(device)
                cnn_out, rnn_out = model(img_in, npy_in)
                val_loss += (criterion(cnn_out, img_t) + criterion(rnn_out, npy_t)).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"   📝 Epoch {epoch+1}: Train Loss={avg_train:.6f}, Val Loss={avg_val:.6f}")
        
        # 저장 및 종료 체크
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("🛑 Early Stopping 발동! 학습을 종료합니다.")
            break
            
    print("\n🎉 학습 완료! 'best_deepfake_model.pt' 파일이 생성되었습니다.")