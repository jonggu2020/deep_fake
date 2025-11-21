# calculate_threshold_engine1.py
# (Engine 1 모델들의 정상 데이터 분포 분석 및 임계값 산출)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# --- 1. 설정 및 경로 (학습 코드와 동일해야 함) ---
CONFIG = {
    'dl_batch_size': 256, # 추론용이라 크게 잡아도 됨
    'rnn_hidden_dim': 128,
    'rnn_layers': 2,
    'rnn_type': "GRU",
    'tab_latent_dim': 64,
    'seq_len': 90,
    'n_features': 5
}

# 경로 (Engine 1 저장 경로)
MODEL_DIR = "./models/engine1"
CSV_FILE_PATH = "./master_summary_v11_cleaned_final.csv"
NPY_DIR = "./2_npy_timeseries"

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 모델 클래스 정의 (학습 코드와 100% 일치해야 함) ---
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
        self.enc = nn.GRU(CONFIG['n_features'], hidden_dim, num_layers, batch_first=True)
        self.dec = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, CONFIG['n_features'])
    def forward(self, x):
        _, h = self.enc(x)
        h_rep = h[-1].unsqueeze(1).repeat(1, CONFIG['seq_len'], 1)
        dec_out, _ = self.dec(h_rep)
        return self.out(dec_out)

# --- 3. 데이터 로드 및 전처리 (저장된 스케일러 사용) ---
def load_inference_data():
    print("📥 데이터 및 스케일러 로드 중...")
    
    # 1. CSV 로드
    df = pd.read_csv(CSV_FILE_PATH)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    feat_cols = [c for c in num_cols if c not in ['label']]
    
    # 2. 저장된 스케일러 로드
    try:
        tab_scaler = load(os.path.join(MODEL_DIR, "tab_scaler.joblib"))
        npy_scaler = load(os.path.join(MODEL_DIR, "npy_scaler.joblib"))
    except FileNotFoundError:
        print("❌ 스케일러 파일을 찾을 수 없습니다. 학습이 먼저 완료되어야 합니다.")
        exit()
        
    # 3. Tabular 변환
    X_tab = tab_scaler.transform(df[feat_cols])
    
    # 4. NPY 로드 및 변환
    X_npy = np.zeros((len(df), CONFIG['seq_len'], CONFIG['n_features']), dtype=np.float32)
    valid_indices = [] # NPY가 실제로 있는 인덱스만 추림
    
    print("📥 NPY 매칭 및 변환 중...")
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
                if curr > CONFIG['seq_len']: m = m[:CONFIG['seq_len']]
                elif curr < CONFIG['seq_len']: m = np.vstack([m, np.zeros((CONFIG['seq_len']-curr, CONFIG['n_features']))])
                
                X_npy[idx] = m
                valid_indices.append(idx)
        except: pass
    
    # NPY 스케일링
    N, T, F = X_npy.shape
    X_npy = npy_scaler.transform(X_npy.reshape(-1, F)).reshape(N, T, F)
    
    # NPY가 있는 데이터만 필터링 (분석 정확도를 위해)
    X_tab = X_tab[valid_indices]
    X_npy = X_npy[valid_indices]
    
    return X_tab, X_npy, len(feat_cols)

# --- 4. 메인 실행: 임계값 계산 ---
if __name__ == "__main__":
    # 1. 데이터 준비
    X_tab, X_npy, tab_dim = load_inference_data()
    print(f"📊 분석 대상 데이터 수: {len(X_tab)}개")

    # 2. 모델 불러오기
    print("\n🔄 모델 로드 중...")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    
    # Tabular AE
    model_tab = TabularAE(tab_dim, CONFIG['tab_latent_dim']).to(device)
    model_tab.load_state_dict(torch.load(os.path.join(MODEL_DIR, "tabular_ae.pth"), map_location=device))
    model_tab.eval()
    
    # RNN AE
    model_rnn = RNNAE(CONFIG['rnn_type'], CONFIG['rnn_hidden_dim'], CONFIG['rnn_layers']).to(device)
    model_rnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "rnn_ae.pth"), map_location=device))
    model_rnn.eval()
    
    # 3. 점수(Loss/Prob) 계산
    print("🔍 각 모델별 Anomaly Score 계산 중...")
    
    # XGBoost Score (Probability of class 1)
    # 주의: 학습 시 class 1을 '이상치(Pseudo-Anomaly)'로 뒀는지 확인 필요.
    # 보통 IsolationForest -1을 1로 뒀으므로, 1에 가까울수록 이상치.
    xgb_probs = xgb_model.predict_proba(X_tab)[:, 1] 
    
    # Deep Learning Scores
    tab_losses = []
    rnn_losses = []
    
    ds = TensorDataset(torch.FloatTensor(X_tab), torch.FloatTensor(X_npy))
    loader = DataLoader(ds, batch_size=CONFIG['dl_batch_size'], shuffle=False)
    
    criterion = nn.MSELoss(reduction='none') # 개별 샘플별 Loss 계산
    
    with torch.no_grad():
        for bx_tab, bx_npy in tqdm(loader):
            bx_tab, bx_npy = bx_tab.to(device), bx_npy.to(device)
            
            # Tabular AE Loss
            out_tab = model_tab(bx_tab)
            loss_t = torch.mean((out_tab - bx_tab)**2, dim=1) # (Batch,)
            tab_losses.extend(loss_t.cpu().numpy())
            
            # RNN AE Loss
            out_rnn = model_rnn(bx_npy)
            loss_r = torch.mean((out_rnn - bx_npy)**2, dim=[1, 2]) # (Batch,)
            rnn_losses.extend(loss_r.cpu().numpy())
            
    tab_losses = np.array(tab_losses)
    rnn_losses = np.array(rnn_losses)
    
    # 4. 통계 및 임계값 제안
    def print_stats(name, data):
        mean, std, dmax = np.mean(data), np.std(data), np.max(data)
        th_3std = mean + 3 * std
        print(f"\n📌 [{name}] Score 통계")
        print(f"   Mean: {mean:.4f} | Std: {std:.4f} | Max: {dmax:.4f}")
        print(f"   👉 추천 임계값 (Mean + 3σ): {th_3std:.4f}")
        return th_3std

    print("\n" + "="*40)
    th_xgb = print_stats("XGBoost (Probability)", xgb_probs)
    th_tab = print_stats("Tabular AE (MSE)", tab_losses)
    th_rnn = print_stats("RNN AE (MSE)", rnn_losses)
    print("="*40)
    
    # 5. 임계값 저장
    thresholds = {
        "xgb_threshold": float(th_xgb),
        "tabular_ae_threshold": float(th_tab),
        "rnn_ae_threshold": float(th_rnn)
    }
    dump(thresholds, os.path.join(MODEL_DIR, "thresholds.joblib"))
    print(f"\n💾 임계값 설정 파일 저장 완료: {os.path.join(MODEL_DIR, 'thresholds.joblib')}")

    # 6. 히스토그램 시각화 및 저장
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(xgb_probs, bins=50, kde=True, color='blue')
    plt.axvline(th_xgb, color='red', linestyle='--', label=f'Threshold: {th_xgb:.2f}')
    plt.title("XGBoost Anomaly Probability")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    sns.histplot(tab_losses, bins=50, kde=True, color='green')
    plt.axvline(th_tab, color='red', linestyle='--', label=f'Threshold: {th_tab:.2f}')
    plt.title("Tabular AE Reconstruction Error")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    sns.histplot(rnn_losses, bins=50, kde=True, color='orange')
    plt.axvline(th_rnn, color='red', linestyle='--', label=f'Threshold: {th_rnn:.2f}')
    plt.title("RNN AE Reconstruction Error")
    plt.legend()
    
    save_img_path = os.path.join(MODEL_DIR, "threshold_distributions.png")
    plt.savefig(save_img_path)
    print(f"📈 분포 그래프 저장 완료: {save_img_path}")
    # plt.show() # 필요시 주석 해제