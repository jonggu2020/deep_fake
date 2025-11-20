# train_integrated.py
# (48GB RAM 활용: 100회 자동 튜닝 버전)

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

# 설정 파일 임포트
from sweep_config_integrated import sweep_config 

# --- 1. 사용자 설정 (경로) ---

# ⚠️ [확인] 본인의 환경에 맞는 파일 경로인지 꼭 확인하세요.
CSV_FILE_PATH = "./cleaned_statistics_all_merged.csv"
NPY_DIR = "./2_npy_timeseries"

# NPY 데이터 설정
NPY_SEQ_LENGTH = 90
NPY_FEATURES = 5

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 데이터 준비 함수 (In-Memory Loading) ---

def load_all_data_to_ram():
    print("="*60)
    print(f"🚀 [RAM 최적화] 데이터를 메모리에 로드합니다...")
    print(f"   - CSV 경로: {CSV_FILE_PATH}")
    print(f"   - NPY 폴더: {NPY_DIR}")
    print("="*60)
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ [오류] CSV 파일을 찾을 수 없습니다: {CSV_FILE_PATH}")
        return None

    try:
        # 1. CSV 로드 및 전처리
        df = pd.read_csv(CSV_FILE_PATH)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        feat_cols = [c for c in num_cols if c not in ['label']]
        
        print(f"✓ CSV 로드 완료: {len(df)} 행")
        
        # 2. Tabular Scaler Fitting
        print("📊 정형 데이터(Tabular) 스케일링 중...")
        tab_scaler = StandardScaler().fit(df[feat_cols])
        X_tab_all = tab_scaler.transform(df[feat_cols]) # (N, Features)
        
        # 3. Pseudo-labeling (XGBoost 학습용)
        print("🌲 Isolation Forest로 가상 라벨(Pseudo-label) 생성 중...")
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        pseudo_labels = (iso.fit_predict(X_tab_all) == -1).astype(int)
        print(f"   - 정상(0): {np.sum(pseudo_labels==0)}, 이상(1): {np.sum(pseudo_labels==1)}")
        
        # 4. NPY 데이터 전량 로드
        print(f"📥 시계열 데이터(NPY) 로드 중...")
        
        # 결과를 담을 빈 배열 생성 (N, 90, 5)
        X_npy_all = np.zeros((len(df), NPY_SEQ_LENGTH, NPY_FEATURES), dtype=np.float32)
        npy_samples_for_scaler = []
        
        missing_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="NPY Loading"):
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
                    
                    # 길이 맞추기 (Padding/Truncating)
                    curr = m.shape[0]
                    if curr > NPY_SEQ_LENGTH:
                        m = m[:NPY_SEQ_LENGTH]
                    elif curr < NPY_SEQ_LENGTH:
                        m = np.vstack([m, np.zeros((NPY_SEQ_LENGTH - curr, NPY_FEATURES))])
                    
                    X_npy_all[idx] = m
                    
                    # 스케일러 학습용 샘플링 (앞쪽 5000개만)
                    if len(npy_samples_for_scaler) < 5000:
                        npy_samples_for_scaler.append(m)
                        
                except Exception:
                    missing_count += 1
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"   ⚠️ {missing_count}개의 NPY 파일 누락 (0으로 대체됨)")
            
        # 5. NPY Scaler Fitting & Transform
        print("📉 시계열 데이터 스케일링 중...")
        if npy_samples_for_scaler:
            npy_scaler = StandardScaler()
            npy_scaler.fit(np.concatenate(npy_samples_for_scaler))
            
            # 전체 데이터 변환
            N, T, F = X_npy_all.shape
            X_npy_flat = X_npy_all.reshape(-1, F)
            X_npy_flat = npy_scaler.transform(X_npy_flat)
            X_npy_all = X_npy_flat.reshape(N, T, F)
        else:
            print("⚠️ NPY 데이터가 충분하지 않아 스케일링을 건너뜁니다.")

        # 6. 학습/검증 데이터 분리
        indices = np.arange(len(df))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=pseudo_labels, random_state=42)
        
        # 최종 데이터 딕셔너리 구성
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

# --- 3. 모델 정의 (Autoencoders) ---

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
        
        # 마지막 Hidden state를 시퀀스 길이만큼 반복하여 디코더 입력으로 사용
        h_rep = h[-1].unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1)
        dec_out, _ = self.dec(h_rep)
        return self.out(dec_out)

# --- 4. 메인 학습 함수 (WandB Agent가 호출) ---

def train_pipeline():
    # WandB 초기화 (Sweep Controller가 설정을 주입)
    wandb.init()
    cfg = wandb.config
    
    # --- Phase 1: XGBoost 학습 ---
    print(f"\n🚀 [Sweep Run] XGBoost 학습: n_estimators={cfg.xgb_n_estimators}, depth={cfg.xgb_max_depth}")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        objective='binary:logistic',
        tree_method='hist', 
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42
    )
    
    # RAM에 있는 데이터 사용
    xgb_model.fit(
        GLOBAL_DATA['train']['tab'], GLOBAL_DATA['train']['y'],
        eval_set=[(GLOBAL_DATA['val']['tab'], GLOBAL_DATA['val']['y'])],
        verbose=False
    )
    
    # XGBoost 평가
    xgb_preds = xgb_model.predict_proba(GLOBAL_DATA['val']['tab'])[:, 1]
    xgb_loss = log_loss(GLOBAL_DATA['val']['y'], xgb_preds)
    xgb_acc = accuracy_score(GLOBAL_DATA['val']['y'], (xgb_preds > 0.5).astype(int))
    
    print(f"   ✅ XGBoost 완료 | Val Loss: {xgb_loss:.4f} | Acc: {xgb_acc:.4f}")
    
    # --- Phase 2: Deep Learning (PyTorch) 학습 ---
    print(f"🚀 [Sweep Run] DL 학습: Tabular({cfg.tab_latent_dim}) + RNN({cfg.rnn_type}/{cfg.rnn_hidden_dim})")
    
    # TensorDataset 변환
    train_ds = TensorDataset(
        torch.FloatTensor(GLOBAL_DATA['train']['tab']),
        torch.FloatTensor(GLOBAL_DATA['train']['npy'])
    )
    val_ds = TensorDataset(
        torch.FloatTensor(GLOBAL_DATA['val']['tab']),
        torch.FloatTensor(GLOBAL_DATA['val']['npy'])
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.dl_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.dl_batch_size, shuffle=False)
    
    # 모델 초기화
    tab_input_dim = GLOBAL_DATA['input_dim']
    model_tab = TabularAE(tab_input_dim, cfg.tab_latent_dim).to(device)
    model_rnn = RNNAE(cfg.rnn_type, cfg.rnn_hidden_dim, cfg.rnn_layers).to(device)
    
    optimizer = optim.Adam(
        list(model_tab.parameters()) + list(model_rnn.parameters()), 
        lr=cfg.dl_learning_rate
    )
    criterion = nn.MSELoss()
    
    # 학습 루프 (15 Epoch)
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
        
        # 종합 점수 (XGBoost Loss + DL Reconstruction Loss)
        global_score = avg_dl_loss + xgb_loss 
        
        wandb.log({
            "epoch": epoch + 1,
            "global_score": global_score,
            "xgb_val_loss": xgb_loss,
            "dl_total_val_loss": avg_dl_loss,
            "ae_tabular_loss": avg_tab_loss,
            "ae_rnn_loss": avg_rnn_loss
        })
        
    print(f"   🏁 학습 종료 | Final Global Score: {global_score:.4f}")

# --- 5. 실행부 ---

if __name__ == "__main__":
    
    # 1. 데이터 로드 (최초 1회 실행)
    # 전역 변수 GLOBAL_DATA에 데이터를 올려두고 100번의 Sweep 동안 계속 재사용합니다.
    GLOBAL_DATA = load_all_data_to_ram()
    
    if GLOBAL_DATA:
        print("\n" + "="*60)
        print("✅ 데이터 준비 완료! WandB Sweep Agent를 시작합니다.")
        print("🚀 총 100회의 하이퍼파라미터 튜닝을 진행합니다.")
        print("="*60 + "\n")
        
        # 2. Sweep 프로젝트 등록
        sweep_id = wandb.sweep(sweep_config, project="deepfake-LQ-Ensemble-RAM")
        
        # 3. Agent 실행 (count=100)
        wandb.agent(sweep_id, function=train_pipeline, count=100)
    else:
        print("❌ 데이터 로드 실패로 프로그램을 종료합니다.")