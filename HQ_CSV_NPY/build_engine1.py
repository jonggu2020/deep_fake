# build_engine1.py
# (엔진 1: XGBoost + Tabular AE + GRU AE 최종 학습 및 저장)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump # 모델/스케일러 저장용
from tqdm import tqdm

# --- 1. 최적 하이퍼파라미터 (사용자 제공) ---
CONFIG = {
    'dl_batch_size': 64,
    'dl_learning_rate': 0.00262, # (반올림)
    'rnn_hidden_dim': 128,
    'rnn_layers': 2,
    'rnn_type': "GRU",
    'tab_latent_dim': 64,
    'xgb_learning_rate': 0.2,
    'xgb_max_depth': 3,
    'xgb_n_estimators': 200
}

# --- 2. 경로 설정 ---
CSV_FILE_PATH = "./master_summary_v11_cleaned_final.csv" # (43,000개 데이터)
NPY_DIR = "./2_npy_timeseries"
MODEL_SAVE_DIR = "./models/engine1" # 저장 경로

# NPY 설정
NPY_SEQ_LENGTH = 90
NPY_FEATURES = 5

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Engine 1 학습 시작 (Device: {device})")

# --- 3. 데이터 로드 (RAM 최적화 버전) ---
def load_data():
    print("📥 데이터 로드 중...")
    df = pd.read_csv(CSV_FILE_PATH)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    feat_cols = [c for c in num_cols if c not in ['label']]
    
    # 1. Tabular Scaler
    print("📊 Tabular Scaler 피팅...")
    tab_scaler = StandardScaler().fit(df[feat_cols])
    X_tab = tab_scaler.transform(df[feat_cols])
    
    # 2. Pseudo-labeling (XGBoost용)
    print("🌲 Isolation Forest 라벨링...")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    y_pseudo = (iso.fit_predict(X_tab) == -1).astype(int)
    
    # 3. NPY 로드
    print("📥 NPY 파일 로드 중...")
    X_npy = np.zeros((len(df), NPY_SEQ_LENGTH, NPY_FEATURES), dtype=np.float32)
    samples_for_scaler = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            path = os.path.join(NPY_DIR, f"{row['video_id']}.npy")
            if os.path.exists(path):
                d = np.load(path, allow_pickle=True).item()
                m = np.stack([
                    d['mouth']['laplacian_mean'], d['mouth']['laplacian_var'],
                    d['mouth']['light_intensity_mean'], d['mouth']['light_intensity_change'],
                    d['mouth']['area']
                ], axis=1)
                
                curr = m.shape[0]
                if curr > NPY_SEQ_LENGTH: m = m[:NPY_SEQ_LENGTH]
                elif curr < NPY_SEQ_LENGTH: m = np.vstack([m, np.zeros((NPY_SEQ_LENGTH-curr, NPY_FEATURES))])
                
                X_npy[idx] = m
                if len(samples_for_scaler) < 5000: samples_for_scaler.append(m)
        except: pass
        
    # 4. NPY Scaler
    print("📉 NPY Scaler 피팅...")
    npy_scaler = StandardScaler()
    if samples_for_scaler:
        npy_scaler.fit(np.concatenate(samples_for_scaler))
        N, T, F = X_npy.shape
        X_npy = npy_scaler.transform(X_npy.reshape(-1, F)).reshape(N, T, F)
        
    return X_tab, X_npy, y_pseudo, tab_scaler, npy_scaler, len(feat_cols)

# --- 4. 모델 클래스 ---
class TabularAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

class RNNAE(nn.Module):
    def __init__(self, rnn_type, hidden_dim, num_layers):
        super().__init__()
        self.enc = nn.GRU(NPY_FEATURES, hidden_dim, num_layers, batch_first=True)
        self.dec = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, NPY_FEATURES)
    def forward(self, x):
        _, h = self.enc(x)
        h_rep = h[-1].unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1)
        dec_out, _ = self.dec(h_rep)
        return self.out(dec_out)

# --- 5. 메인 실행 ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    # 1. 데이터 준비
    X_tab, X_npy, y, tab_scaler, npy_scaler, tab_dim = load_data()
    
    # 2. XGBoost 학습 및 저장
    print("\n🚀 [1/3] XGBoost 학습 시작...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=CONFIG['xgb_n_estimators'],
        max_depth=CONFIG['xgb_max_depth'],
        learning_rate=CONFIG['xgb_learning_rate'],
        tree_method='hist', device="cuda", random_state=42
    )
    xgb_model.fit(X_tab, y)
    xgb_model.save_model(os.path.join(MODEL_SAVE_DIR, "xgb_model.json"))
    print("✅ XGBoost 저장 완료.")
    
    # 3. PyTorch 모델 학습
    print("\n🚀 [2/3] PyTorch Deep Learning 학습 시작...")
    
    ds = TensorDataset(torch.FloatTensor(X_tab), torch.FloatTensor(X_npy))
    loader = DataLoader(ds, batch_size=CONFIG['dl_batch_size'], shuffle=True)
    
    model_tab = TabularAE(tab_dim, CONFIG['tab_latent_dim']).to(device)
    model_rnn = RNNAE(CONFIG['rnn_type'], CONFIG['rnn_hidden_dim'], CONFIG['rnn_layers']).to(device)
    
    optimizer = optim.Adam(list(model_tab.parameters()) + list(model_rnn.parameters()), lr=CONFIG['dl_learning_rate'])
    criterion = nn.MSELoss()
    
    epochs = 20 # (최종 학습이므로 충분히)
    model_tab.train(); model_rnn.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for bx_tab, bx_npy in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            bx_tab, bx_npy = bx_tab.to(device), bx_npy.to(device)
            optimizer.zero_grad()
            loss = criterion(model_tab(bx_tab), bx_tab) + criterion(model_rnn(bx_npy), bx_npy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Loss: {total_loss/len(loader):.4f}")
        
    # 4. PyTorch 모델 및 Scaler 저장
    torch.save(model_tab.state_dict(), os.path.join(MODEL_SAVE_DIR, "tabular_ae.pth"))
    torch.save(model_rnn.state_dict(), os.path.join(MODEL_SAVE_DIR, "rnn_ae.pth"))
    dump(tab_scaler, os.path.join(MODEL_SAVE_DIR, "tab_scaler.joblib"))
    dump(npy_scaler, os.path.join(MODEL_SAVE_DIR, "npy_scaler.joblib"))
    
    print("\n🎉 Engine 1 (영상/통계) 구축 완료! 모든 파일이 저장되었습니다.")
    print(f"📂 저장 위치: {MODEL_SAVE_DIR}")