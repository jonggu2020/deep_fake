# train_integrated_ram.py
# (48GB RAM 활용: 모든 데이터를 메모리에 로드하여 초고속 학습)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import wandb
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm

# 설정 파일 임포트 (기존 파일 사용)
from sweep_config_integrated import sweep_config 

# --- 1. 사용자 설정 (경로) ---

# ⚠️ [수정필요 1] 43,000개 원본 CSV 파일 경로
CSV_FILE_PATH = "./master_summary_v11_cleaned_final.csv"

# ⚠️ [수정필요 2] 43,000개 NPY 파일 폴더
NPY_DIR = "./2_npy_timeseries"

# NPY 설정
NPY_SEQ_LENGTH = 90
NPY_FEATURES = 5

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 PyTorch Device: {device}")

# --- 2. 데이터 준비 함수 (In-Memory Loading) ---

def load_all_data_to_ram():
    print("="*50)
    print("🚀 [RAM 최적화] 모든 데이터를 메모리로 로드합니다...")
    print("="*50)
    
    try:
        # 1. CSV 로드 및 전처리
        df = pd.read_csv(CSV_FILE_PATH)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        feat_cols = [c for c in num_cols if c not in ['label']]
        
        print(f"✓ CSV 로드 완료: {len(df)} 행")
        
        # 2. CSV Scaler Fitting
        print("📊 Tabular Scaler 피팅 중...")
        tab_scaler = StandardScaler().fit(df[feat_cols])
        X_tab_all = tab_scaler.transform(df[feat_cols]) # (N, 120) Numpy 배열
        
        # 3. Pseudo-labeling (XGBoost용)
        print("🌲 Isolation Forest로 Pseudo-label 생성 중...")
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        pseudo_labels = (iso.fit_predict(X_tab_all) == -1).astype(int)
        print(f"   - 정상(0): {np.sum(pseudo_labels==0)}, 의심(1): {np.sum(pseudo_labels==1)}")
        
        # 4. NPY 데이터 전량 로드 (핵심 최적화)
        print(f"📥 43,000개 NPY 파일 로드 중 (RAM: 48GB 충분)...")
        
        # 결과를 담을 빈 배열 생성 (N, 90, 5)
        X_npy_all = np.zeros((len(df), NPY_SEQ_LENGTH, NPY_FEATURES), dtype=np.float32)
        
        # NPY Scaler를 위한 샘플 데이터 수집
        npy_samples_for_scaler = []
        
        # 파일 로딩 루프
        missing_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="NPY 로딩"):
            video_id = row['video_id']
            npy_path = os.path.join(NPY_DIR, f"{video_id}.npy")
            
            if os.path.exists(npy_path):
                try:
                    d = np.load(npy_path, allow_pickle=True).item()
                    # 'mouth' 특징 추출
                    m = np.stack([
                        d['mouth']['laplacian_mean'], d['mouth']['laplacian_var'],
                        d['mouth']['light_intensity_mean'], d['mouth']['light_intensity_change'],
                        d['mouth']['area']
                    ], axis=1) # (T, 5)
                    
                    # Pad/Truncate (길이 맞추기)
                    curr = m.shape[0]
                    if curr > NPY_SEQ_LENGTH:
                        m = m[:NPY_SEQ_LENGTH]
                    elif curr < NPY_SEQ_LENGTH:
                        m = np.vstack([m, np.zeros((NPY_SEQ_LENGTH - curr, NPY_FEATURES))])
                    
                    X_npy_all[idx] = m
                    
                    # 스케일러 학습용 샘플링 (처음 5000개만 사용)
                    if len(npy_samples_for_scaler) < 5000:
                        npy_samples_for_scaler.append(m)
                        
                except Exception:
                    missing_count += 1
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"   ⚠️ {missing_count}개의 NPY 파일을 찾을 수 없거나 오류가 있어 0으로 채웠습니다.")
            
        # 5. NPY Scaler Fitting & Transform
        print("📉 NPY 데이터 스케일링 중...")
        if npy_samples_for_scaler:
            # 스케일러 피팅
            npy_scaler = StandardScaler()
            npy_scaler.fit(np.concatenate(npy_samples_for_scaler))
            
            # 전체 데이터 변환 (Batch 처리로 메모리 효율화)
            # (N, 90, 5) -> (N*90, 5) -> transform -> (N, 90, 5)
            N, T, F = X_npy_all.shape
            X_npy_flat = X_npy_all.reshape(-1, F)
            X_npy_flat = npy_scaler.transform(X_npy_flat)
            X_npy_all = X_npy_flat.reshape(N, T, F)
        else:
            print("⚠️ NPY 데이터가 없어 스케일링을 건너뜁니다.")

        # 6. 학습/검증 분리 (Indices)
        indices = np.arange(len(df))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=pseudo_labels, random_state=42)
        
        # 최종 데이터 딕셔너리 반환
        data = {
            "train": {
                "tab": X_tab_all[train_idx],
                "npy": X_npy_all[train_idx],
                "y": pseudo_labels[train_idx]
            },
            "val": {
                "tab": X_tab_all[val_idx],
                "npy": X_npy_all[val_idx],
                "y": pseudo_labels[val_idx]
            },
            "input_dim": len(feat_cols)
        }
        
        print("✓ 모든 데이터 준비 완료.")
        return data

    except Exception as e:
        print(f"❌ 데이터 로드 중 치명적 오류: {e}")
        return None

# --- 3. PyTorch 모델 정의 ---

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

# --- 4. 메인 학습 함수 (WandB Agent) ---

def train_pipeline():
    wandb.init()
    cfg = wandb.config
    
    # --- Phase 1: XGBoost 학습 ---
    print("\n🚀 [Phase 1] XGBoost 학습 (RAM)")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        objective='binary:logistic',
        tree_method='hist', 
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42
    )
    
    # 메모리에 있는 데이터 바로 사용
    xgb_model.fit(
        GLOBAL_DATA['train']['tab'], GLOBAL_DATA['train']['y'],
        eval_set=[(GLOBAL_DATA['val']['tab'], GLOBAL_DATA['val']['y'])],
        verbose=False
    )
    
    # 평가
    xgb_preds = xgb_model.predict_proba(GLOBAL_DATA['val']['tab'])[:, 1]
    xgb_loss = log_loss(GLOBAL_DATA['val']['y'], xgb_preds)
    xgb_acc = accuracy_score(GLOBAL_DATA['val']['y'], (xgb_preds > 0.5).astype(int))
    
    print(f"   ✅ XGBoost 완료 | Val Loss: {xgb_loss:.4f} | Acc: {xgb_acc:.4f}")
    
    # --- Phase 2: PyTorch 학습 ---
    print("\n🚀 [Phase 2] PyTorch 학습 (RAM)")
    
    # TensorDataset으로 변환 (초고속)
    # (메모리에 있는 numpy 배열을 그대로 Tensor로 변환)
    train_ds = TensorDataset(
        torch.FloatTensor(GLOBAL_DATA['train']['tab']),
        torch.FloatTensor(GLOBAL_DATA['train']['npy'])
    )
    val_ds = TensorDataset(
        torch.FloatTensor(GLOBAL_DATA['val']['tab']),
        torch.FloatTensor(GLOBAL_DATA['val']['npy'])
    )
    
    # DataLoader (num_workers=0 권장: 이미 메모리에 있어서 멀티프로세싱 불필요)
    train_loader = DataLoader(train_ds, batch_size=cfg.dl_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.dl_batch_size, shuffle=False)
    
    # 모델 설정
    tab_input_dim = GLOBAL_DATA['input_dim']
    model_tab = TabularAE(tab_input_dim, cfg.tab_latent_dim).to(device)
    model_rnn = RNNAE(cfg.rnn_type, cfg.rnn_hidden_dim, cfg.rnn_layers).to(device)
    
    optimizer = optim.Adam(list(model_tab.parameters()) + list(model_rnn.parameters()), lr=cfg.dl_learning_rate)
    criterion = nn.MSELoss()
    
    # 학습 루프
    epochs = 15
    
    for epoch in range(epochs):
        model_tab.train(); model_rnn.train()
        
        for tab_x, npy_x in train_loader:
            tab_x, npy_x = tab_x.to(device), npy_x.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model_tab(tab_x), tab_x) + criterion(model_rnn(npy_x), npy_x)
            loss.backward()
            optimizer.step()
            
        # Validation
        model_tab.eval(); model_rnn.eval()
        val_loss_sum = 0; val_tab_sum = 0; val_rnn_sum = 0
        
        with torch.no_grad():
            for tab_x, npy_x in val_loader:
                tab_x, npy_x = tab_x.to(device), npy_x.to(device)
                l_tab = criterion(model_tab(tab_x), tab_x)
                l_rnn = criterion(model_rnn(npy_x), npy_x)
                
                val_loss_sum += (l_tab + l_rnn).item()
                val_tab_sum += l_tab.item()
                val_rnn_sum += l_rnn.item()
                
        avg_dl_loss = val_loss_sum / len(val_loader)
        avg_tab_loss = val_tab_sum / len(val_loader)
        avg_rnn_loss = val_rnn_sum / len(val_loader)
        
        # Global Score
        global_score = avg_dl_loss + xgb_loss 
        
        wandb.log({
            "epoch": epoch + 1,
            "global_score": global_score,
            "xgb_val_loss": xgb_loss,
            "dl_total_val_loss": avg_dl_loss,
            "ae_tabular_loss": avg_tab_loss,
            "ae_rnn_loss": avg_rnn_loss
        })
        
        print(f"   Epoch {epoch+1} | Global: {global_score:.4f} (XGB: {xgb_loss:.4f} + DL: {avg_dl_loss:.4f})")

# --- 5. 실행 ---

if __name__ == "__main__":
    
    # [중요] 전역 변수에 데이터 로드 (Sweep 실행 시 재로딩 방지)
    # 48GB RAM이 있으므로 전역 변수에 한 번만 올려두고 계속 재사용합니다.
    GLOBAL_DATA = load_all_data_to_ram()
    
    if GLOBAL_DATA:
        print("\n✅ 데이터 로드 완료. WandB Agent를 시작합니다.")
        
        sweep_id = wandb.sweep(sweep_config, project="deepfake-Integrated-Ensemble-RAM")
        wandb.agent(sweep_id, function=train_pipeline, count=15)