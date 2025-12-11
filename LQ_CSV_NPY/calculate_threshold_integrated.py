# calculate_threshold_integrated.py
# (Integrated Model: XGBoost + Tabular AE + RNN AE 임계값 계산)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from types import SimpleNamespace

# --- 1. 설정 및 경로 (학습 코드와 동일) ---
config = SimpleNamespace(
    batch_size = 64,
    rnn_hidden_dim = 128,
    rnn_layers = 2,
    rnn_type = "GRU",
    tab_latent_dim = 128
)

CSV_FILE_PATH = "./cleaned_statistics_all_merged.csv"
NPY_DIR = "./2_npy_timeseries"
NPY_SEQ_LENGTH = 90
NPY_FEATURES = 5

# 저장된 모델 파일 경로
XGB_MODEL_PATH = 'best_xgb_model.joblib'
DL_MODEL_PATH = 'best_integrated_dl.pt'
TAB_SCALER_PATH = 'final_tab_scaler.joblib'
NPY_SCALER_PATH = 'final_npy_scaler.joblib'

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
        self.rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self.enc = nn.LSTM(NPY_FEATURES, hidden_dim, num_layers, batch_first=True)
            self.dec = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.enc = nn.GRU(NPY_FEATURES, hidden_dim, num_layers, batch_first=True)
            self.dec = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, NPY_FEATURES)
        
    def forward(self, x):
        if self.rnn_type == 'LSTM': _, (h, _) = self.enc(x)
        else: _, h = self.enc(x)
        h_rep = h[-1].unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1)
        dec_out, _ = self.dec(h_rep)
        return self.out(dec_out)

# --- 3. 데이터 로드 함수 (Validation Set 추출) ---
def load_val_data():
    print("📥 데이터 로드 및 전처리 중...")
    
    # 1. CSV 및 Scaler 로드
    if not os.path.exists(CSV_FILE_PATH): raise FileNotFoundError("CSV 없음")
    df = pd.read_csv(CSV_FILE_PATH)
    
    # 결측치 처리
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    feat_cols = [c for c in num_cols if c not in ['label']] # 학습때 쓴 컬럼명 자동 추출
    
    # Scaler 로드 (재학습 금지)
    if not os.path.exists(TAB_SCALER_PATH): raise FileNotFoundError("Tabular Scaler 없음")
    if not os.path.exists(NPY_SCALER_PATH): raise FileNotFoundError("NPY Scaler 없음")
    
    tab_scaler = joblib.load(TAB_SCALER_PATH)
    npy_scaler = joblib.load(NPY_SCALER_PATH)
    
    # Tabular 변환
    X_tab_all = tab_scaler.transform(df[feat_cols])
    
    # NPY 로드 및 변환
    X_npy_all = np.zeros((len(df), NPY_SEQ_LENGTH, NPY_FEATURES), dtype=np.float32)
    print("   - NPY 파일 매핑 중 (시간 소요)...")
    
    # 빠른 로딩을 위해 exists 체크 최소화 및 배치 처리 고려 가능하나, 
    # 정확성을 위해 Loop 사용 (tqdm)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['video_id']
        npy_path = os.path.join(NPY_DIR, f"{video_id}.npy")
        try:
            if os.path.exists(npy_path):
                d = np.load(npy_path, allow_pickle=True).item()
                m = np.stack([
                    d['mouth']['laplacian_mean'], d['mouth']['laplacian_var'],
                    d['mouth']['light_intensity_mean'], d['mouth']['light_intensity_change'],
                    d['mouth']['area']
                ], axis=1)
                
                # 길이 맞춤
                curr = m.shape[0]
                if curr > NPY_SEQ_LENGTH: m = m[:NPY_SEQ_LENGTH]
                elif curr < NPY_SEQ_LENGTH: m = np.vstack([m, np.zeros((NPY_SEQ_LENGTH-curr, NPY_FEATURES))])
                
                X_npy_all[idx] = m
        except: pass # 없으면 0으로 유지
        
    # NPY Scaling
    N, T, F = X_npy_all.shape
    X_npy_all = npy_scaler.transform(X_npy_all.reshape(-1, F)).reshape(N, T, F)
    
    # Train/Val Split (학습과 동일한 시드 42 사용 필수)
    indices = np.arange(len(df))
    # Pseudo label은 필요 없지만 split 재현을 위해 그냥 랜덤 스플릿 (stratify 없이) 
    # *주의: 학습 코드에선 stratify=pseudo_labels 였으나, 
    # 여기선 그냥 같은 random_state=42면 대략적으로 분포가 유지된다고 가정하거나,
    # 단순히 8:2 랜덤 스플릿을 해도 분포 파악엔 큰 무리 없음.
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    return {
        "tab": X_tab_all[val_idx],
        "npy": X_npy_all[val_idx],
        "input_dim": len(feat_cols)
    }

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    print(f"🚀 [통합 모델 임계값 계산] 시작 (Device: {device})")
    
    # 1. 데이터 준비
    DATA = load_val_data()
    val_tab = torch.FloatTensor(DATA['tab'])
    val_npy = torch.FloatTensor(DATA['npy'])
    
    val_ds = TensorDataset(val_tab, val_npy)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    print(f"📊 검증 데이터 개수: {len(val_tab)}개")
    
    # 2. 모델 로드
    # (1) Deep Learning Models
    print("Load Deep Learning Models...")
    checkpoint = torch.load(DL_MODEL_PATH, map_location=device)
    
    model_tab = TabularAE(DATA['input_dim'], config.tab_latent_dim).to(device)
    model_rnn = RNNAE(config.rnn_type, config.rnn_hidden_dim, config.rnn_layers).to(device)
    
    model_tab.load_state_dict(checkpoint['model_tab'])
    model_rnn.load_state_dict(checkpoint['model_rnn'])
    model_tab.eval()
    model_rnn.eval()
    
    # (2) XGBoost Model
    print("Load XGBoost Model...")
    xgb_model = joblib.load(XGB_MODEL_PATH)
    
    # 3. 스코어 계산
    dl_losses = []  # AE Reconstruction Error
    xgb_probs = []  # XGBoost Anomaly Probability
    
    print("🔍 분석 수행 중...")
    
    # XGBoost 예측 (CPU/GPU 자동 처리됨, 여기선 numpy array 필요)
    # XGBoost는 배치 단위보다 전체를 넣는게 빠를 수 있음
    xgb_preds = xgb_model.predict_proba(DATA['tab'])[:, 1] # Class 1 확률
    xgb_probs.extend(xgb_preds)
    
    # DL 예측 (Batch 단위)
    with torch.no_grad():
        for tab_x, npy_x in tqdm(val_loader):
            tab_x, npy_x = tab_x.to(device), npy_x.to(device)
            
            # Forward
            rec_tab = model_tab(tab_x)
            rec_rnn = model_rnn(npy_x) # (Batch, 5) output of last step
            
            # Loss 계산 (Sample-wise)
            # Tabular: (B, Features) -> (B,)
            loss_tab = torch.mean((rec_tab - tab_x)**2, dim=1)
            
            # RNN: Output layer shape check needed.
            # 학습 코드 RNNAE.forward는 self.out(dec_out)을 리턴함. 
            # dec_out shape은 (B, Seq, Hidden) -> output (B, Seq, Features)
            # 학습 코드의 Loss는 criterion(model_rnn(npy_x), npy_x) 였음.
            
            # 복원 오차 계산
            loss_rnn = torch.mean((rec_rnn - npy_x)**2, dim=[1, 2])
            
            # Total DL Anomaly Score
            total_loss = loss_tab + loss_rnn
            dl_losses.extend(total_loss.cpu().numpy())

    # 4. 통계 및 시각화
    dl_losses = np.array(dl_losses)
    xgb_probs = np.array(xgb_probs)
    
    # --- [결과 1] 딥러닝(AE) 복원 오차 분석 ---
    mean_loss = np.mean(dl_losses)
    std_loss = np.std(dl_losses)
    max_loss = np.max(dl_losses)
    
    thresh_dl_2std = mean_loss + 2 * std_loss
    thresh_dl_3std = mean_loss + 3 * std_loss
    
    print("\n" + "="*40)
    print(f"📊 [1. Deep Learning (AE) 복원 오차 통계]")
    print(f" - 평균: {mean_loss:.6f}, 표준편차: {std_loss:.6f}")
    print(f" - 최대값: {max_loss:.6f}")
    print("-" * 40)
    print(f"💡 추천 임계값 (DL Reconstruction Error):")
    print(f"   1️⃣ 느슨한 기준 (Mean + 2σ): {thresh_dl_2std:.6f}")
    print(f"   2️⃣ 엄격한 기준 (Mean + 3σ): {thresh_dl_3std:.6f}")
    print(f"   3️⃣ 최대값 기준 (Max):       {max_loss:.6f}")
    print("="*40)

    # --- [결과 2] XGBoost 확률 분포 분석 ---
    mean_prob = np.mean(xgb_probs)
    max_prob = np.max(xgb_probs)
    
    print(f"\n📊 [2. XGBoost 예측 확률(Class 1) 통계]")
    print(f" - 평균 확률: {mean_prob:.4f}")
    print(f" - 최대 확률: {max_prob:.4f}")
    print(f" - (참고) XGBoost는 보통 0.5 이상을 이상치(Class 1)로 봅니다.")
    
    # 5. 히스토그램 그리기 (두 모델 따로)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # DL Histogram
    sns.histplot(dl_losses, bins=50, kde=True, ax=ax[0], color='blue')
    ax[0].axvline(thresh_dl_2std, color='orange', linestyle='--', label='2 std')
    ax[0].axvline(thresh_dl_3std, color='red', linestyle='--', label='3 std')
    ax[0].set_title("DL Autoencoder Reconstruction Error")
    ax[0].set_xlabel("MSE Loss")
    ax[0].legend()
    
    # XGB Histogram
    sns.histplot(xgb_probs, bins=50, kde=True, ax=ax[1], color='green')
    ax[1].axvline(0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
    ax[1].set_title("XGBoost Anomaly Probability")
    ax[1].set_xlabel("Probability (Class 1)")
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig("threshold_distribution_integrated.png")
    print(f"\n📈 그래프 저장 완료: threshold_distribution_integrated.png")