# train_best_integrated.py
# (XGBoost + Tabular AE + RNN AE 최종 학습 및 저장)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
import joblib # 모델 및 스케일러 저장용
from types import SimpleNamespace

# --- 1. 최적의 하이퍼파라미터 (사용자 발굴 + LR 조정) ---
config = SimpleNamespace(
    # Deep Learning params
    dl_batch_size = 64,
    dl_learning_rate = 0.001, # ⚠️ 0.0044 -> 0.001로 하향 조정 (안정성 확보)
    
    # RNN params
    rnn_hidden_dim = 128,
    rnn_layers = 2,
    rnn_type = "GRU",
    
    # Tabular params
    tab_latent_dim = 128,
    
    # XGBoost params
    xgb_learning_rate = 0.01,
    xgb_max_depth = 7,
    xgb_n_estimators = 300,
    
    # Training settings
    epochs = 100 # 충분히 학습
)

# --- 2. 경로 및 설정 ---
CSV_FILE_PATH = "./cleaned_statistics_all_merged.csv"
NPY_DIR = "./2_npy_timeseries"
NPY_SEQ_LENGTH = 90
NPY_FEATURES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(42)

# --- 3. 데이터 로드 및 스케일러 저장 ---
def prepare_data_and_save_scalers():
    print("="*60)
    print("📥 [최종] 데이터 로드 및 전처리 (Scaler 저장 포함)")
    
    if not os.path.exists(CSV_FILE_PATH):
        print("❌ CSV 파일을 찾을 수 없습니다."); exit()

    # 1. CSV 로드
    df = pd.read_csv(CSV_FILE_PATH)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    feat_cols = [c for c in num_cols if c not in ['label']]
    
    # 2. Tabular Scaler 학습 및 저장
    print("💾 Tabular Scaler 저장 중...")
    tab_scaler = StandardScaler().fit(df[feat_cols])
    joblib.dump(tab_scaler, 'final_tab_scaler.joblib') # 저장!
    X_tab_all = tab_scaler.transform(df[feat_cols])
    
    # 3. Pseudo-labeling
    print("🌲 Pseudo-labeling 생성 중...")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    pseudo_labels = (iso.fit_predict(X_tab_all) == -1).astype(int)
    
    # 4. NPY 로드
    print("📥 NPY 데이터 로드 중...")
    X_npy_all = np.zeros((len(df), NPY_SEQ_LENGTH, NPY_FEATURES), dtype=np.float32)
    npy_samples = []
    
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
                
                curr = m.shape[0]
                if curr > NPY_SEQ_LENGTH: m = m[:NPY_SEQ_LENGTH]
                elif curr < NPY_SEQ_LENGTH: m = np.vstack([m, np.zeros((NPY_SEQ_LENGTH-curr, NPY_FEATURES))])
                
                X_npy_all[idx] = m
                if len(npy_samples) < 5000: npy_samples.append(m)
        except: pass

    # 5. NPY Scaler 학습 및 저장
    print("💾 NPY Scaler 저장 중...")
    npy_scaler = StandardScaler()
    if npy_samples:
        npy_scaler.fit(np.concatenate(npy_samples))
        joblib.dump(npy_scaler, 'final_npy_scaler.joblib') # 저장!
        
        N, T, F = X_npy_all.shape
        X_npy_all = npy_scaler.transform(X_npy_all.reshape(-1, F)).reshape(N, T, F)
    
    # 분할
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=pseudo_labels, random_state=42)
    
    return {
        "train": {"tab": X_tab_all[train_idx], "npy": X_npy_all[train_idx], "y": pseudo_labels[train_idx]},
        "val": {"tab": X_tab_all[val_idx], "npy": X_npy_all[val_idx], "y": pseudo_labels[val_idx]},
        "input_dim": len(feat_cols)
    }

# --- 4. 모델 정의 ---
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

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, path='best_integrated_dl.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
    def __call__(self, val_loss, model_tab, model_rnn):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_tab, model_rnn)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'   ⚠️ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_tab, model_rnn)
            self.counter = 0
    def save_checkpoint(self, val_loss, model_tab, model_rnn):
        if self.verbose: print(f'   ✅ Loss decreased ({val_loss:.6f}). Saving models...')
        torch.save({
            'model_tab': model_tab.state_dict(),
            'model_rnn': model_rnn.state_dict()
        }, self.path)

# --- 5. 메인 실행 ---
if __name__ == "__main__":
    # 데이터 로드
    DATA = prepare_data_and_save_scalers()
    
    # --- Phase 1: XGBoost 학습 및 저장 ---
    print("\n🚀 [Phase 1] XGBoost Final Training...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        tree_method='hist', device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42
    )
    xgb_model.fit(
        DATA['train']['tab'], DATA['train']['y'],
        eval_set=[(DATA['val']['tab'], DATA['val']['y'])], verbose=False
    )
    # XGBoost 저장
    joblib.dump(xgb_model, 'best_xgb_model.joblib')
    print("💾 best_xgb_model.joblib 저장 완료!")
    
    val_probs = xgb_model.predict_proba(DATA['val']['tab'])[:, 1]
    xgb_loss = log_loss(DATA['val']['y'], val_probs)
    print(f"   ✅ XGBoost Final Val Loss: {xgb_loss:.4f}")

    # --- Phase 2: PyTorch DL 학습 ---
    print("\n🚀 [Phase 2] Deep Learning Final Training...")
    
    train_ds = TensorDataset(torch.FloatTensor(DATA['train']['tab']), torch.FloatTensor(DATA['train']['npy']))
    val_ds = TensorDataset(torch.FloatTensor(DATA['val']['tab']), torch.FloatTensor(DATA['val']['npy']))
    
    train_loader = DataLoader(train_ds, batch_size=config.dl_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.dl_batch_size, shuffle=False)
    
    model_tab = TabularAE(DATA['input_dim'], config.tab_latent_dim).to(device)
    model_rnn = RNNAE(config.rnn_type, config.rnn_hidden_dim, config.rnn_layers).to(device)
    
    optimizer = optim.Adam(list(model_tab.parameters()) + list(model_rnn.parameters()), lr=config.dl_learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=15, path='best_integrated_dl.pt')
    
    for epoch in range(config.epochs):
        model_tab.train(); model_rnn.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for tab_x, npy_x in loop:
            tab_x, npy_x = tab_x.to(device), npy_x.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model_tab(tab_x), tab_x) + criterion(model_rnn(npy_x), npy_x)
            loss.backward()
            
            # ★ [핵심] Gradient Clipping (안전벨트)
            torch.nn.utils.clip_grad_norm_(model_tab.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_rnn.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # 검증
        model_tab.eval(); model_rnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tab_x, npy_x in val_loader:
                tab_x, npy_x = tab_x.to(device), npy_x.to(device)
                loss = criterion(model_tab(tab_x), tab_x) + criterion(model_rnn(npy_x), npy_x)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"   📝 Epoch {epoch+1}: DL Val Loss = {avg_val_loss:.6f} (XGB: {xgb_loss:.4f})")
        
        early_stopping(avg_val_loss, model_tab, model_rnn)
        if early_stopping.early_stop:
            print("🛑 Early Stopping!")
            break
            
    print("\n🎉 모든 학습이 완료되었습니다.")
    print("📁 생성된 파일 목록:")
    print("   1. best_xgb_model.joblib (XGBoost 모델)")
    print("   2. best_integrated_dl.pt (PyTorch 딥러닝 모델)")
    print("   3. final_tab_scaler.joblib (정형 데이터 스케일러)")
    print("   4. final_npy_scaler.joblib (시계열 데이터 스케일러)")