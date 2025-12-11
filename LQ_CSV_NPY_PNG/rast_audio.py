import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.utils import make_grid  # <--- [추가됨] 그리드 이미지 생성을 위한 함수
import wandb
from tqdm import tqdm
import random

# ============================================================
# 0. 최종 학습 설정 (Final Hyperparameters)
# ============================================================
CONFIG = {
    "project_name": "audio-deepfake-final-training",  # 새로운 프로젝트 이름
    "run_name": "efficientnet_b0_final_run",          # 이번 학습의 이름
    
    # --- 데이터 및 모델 설정 ---
    "data_dir": "./3_audio_spectrograms",
    "image_size": 128,
    "silence_threshold": 10,
    "model_name": "efficientnet_b0",
    "latent_dim": 256,           # Phase 1 최적값
    
    # --- 학습 파라미터 ---
    "batch_size": 16,            # Phase 1 최적값
    "num_epochs": 100,           # 요청사항: 100 에포크로 증가
    "learning_rate": 0.0005,     # 요청사항: 기존(0.0039)보다 낮게 설정하여 정밀 학습 유도
    "optimizer": "adamw",        # Phase 1 최적값
    "weight_decay": 1.28e-4,     # Phase 1 최적값 (0.000128...)
    
    # --- 시스템 설정 ---
    "num_workers": 4,
    "seed": 42
}

# ============================================================
# 1. 유틸리티: 시드 고정 및 디바이스 설정
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Dataset Class
# ============================================================
class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, transform=None, silence_threshold=10):
        self.data_dir = data_dir
        self.transform = transform
        self.silence_threshold = silence_threshold
        
        self.image_paths = []
        # 경로 존재 여부 확인
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ 데이터 폴더를 찾을 수 없습니다: {data_dir}")

        for fname in os.listdir(data_dir):
            if fname.endswith('.png'):
                fpath = os.path.join(data_dir, fname)
                if not self._is_silence(fpath):
                    self.image_paths.append(fpath)
        
        print(f"✅ 데이터셋 로드 완료: 총 {len(self.image_paths)}장 (제외된 침묵 데이터 포함)")
    
    def _is_silence(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return True
        mean_intensity = np.mean(img)
        return mean_intensity < self.silence_threshold
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image

# ============================================================
# 3. Model Architecture: EfficientNet-B0 Autoencoder
# ============================================================
class EfficientNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # --- Encoder (Pre-trained EfficientNet-B0) ---
        print(f"🏗️ 모델 생성 중: EfficientNet-B0 (Latent Dim: {latent_dim})")
        # weights 파라미터 경고를 피하기 위해 최신 방식 권장되지만, 
        # 호환성을 위해 pretrained=True 유지 (경고는 무시해도 학습엔 지장 없음)
        efficientnet = models.efficientnet_b0(pretrained=True)
        
        # EfficientNet의 특징 추출기 부분만 사용
        self.encoder_features = efficientnet.features
        
        # Flatten 및 Latent Vector 생성
        self.encoder_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, latent_dim), # EfficientNet-B0의 마지막 채널은 1280
            nn.ReLU()
        )
        
        # --- Decoder ---
        # Latent Vector를 다시 공간적 특징맵으로 확장
        self.decoder_input = nn.Linear(latent_dim, 1280 * 4 * 4)
        
        self.decoder_layers = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (1280, 4, 4)), # (Batch, 1280, 4, 4)
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x64 -> 128x128 (Output Size)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # 픽셀 값을 0~1 사이로 정규화 (이미지 복원)
        )
    
    def forward(self, x):
        # Encode
        x = self.encoder_features(x)
        latent = self.encoder_head(x)
        
        # Decode
        x = self.decoder_input(latent)
        reconstructed = self.decoder_layers(x)
        
        return reconstructed

# ============================================================
# 4. Training Loop (Final)
# ============================================================
def train_final_model():
    # 1. 시드 설정
    set_seed(CONFIG['seed'])
    
    # 2. WandB 초기화
    wandb.init(
        project=CONFIG['project_name'],
        name=CONFIG['run_name'],
        config=CONFIG,
        reinit=True
    )
    config = wandb.config
    
    # 3. 데이터 로드
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(), # 0~1 사이 값으로 변환됨
    ])
    
    full_dataset = MelSpectrogramDataset(
        data_dir=config.data_dir,
        transform=transform,
        silence_threshold=config.silence_threshold
    )
    
    # Train/Val 분할 (9:1)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    print(f"📊 학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")
    
    # 4. 모델 및 학습 도구 설정
    model = EfficientNetAutoencoder(latent_dim=config.latent_dim).to(device)
    
    criterion = nn.MSELoss() # 복원 오차 (Mean Squared Error)
    
    # Optimizer: AdamW 사용 (Phase 1 최적 결과)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Scheduler: verbose=True 제거 (이전 에러 수정됨)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # WandB에 모델 구조 추적
    wandb.watch(model, log='all', log_freq=100)
    
    best_val_loss = float('inf')
    
    # 5. 학습 루프 시작
    print("\n🚀 최종 학습 시작 (100 Epochs)...")
    
    for epoch in range(config.num_epochs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for images in train_pbar:
            images = images.to(device)
            
            # Forward
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        sample_images = None # 초기화
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
            for i, images in enumerate(val_pbar):
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.6f}"})
                
                # 첫 번째 배치의 첫 4장 이미지만 저장 (시각화용)
                if i == 0:
                    # Tensor -> Numpy 변환 및 시각화 준비
                    orig = images[:4].cpu()
                    recon = reconstructed[:4].cpu()
                    
                    # dim=2(Height)로 붙였으므로 (B, C, H*2, W) 형태가 됨.
                    # 즉, 위(원본), 아래(복원) 형태의 세로로 긴 이미지들이 배치로 묶임.
                    comparison = torch.cat([orig, recon], dim=2) 
                    sample_images = comparison
        
        avg_val_loss = val_loss / len(val_loader)
        
        # --- Logging & Saving ---
        current_lr = optimizer.param_groups[0]['lr']
        
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr
        }
        
        # 10 에포크마다 이미지 시각화 로깅 (원본 vs 복원)
        # sample_images가 None이 아닐 때만 로깅
        if sample_images is not None and ((epoch + 1) % 10 == 0 or (epoch + 1) == 1):
            # [수정됨] 4D Tensor (Batch, C, H, W) -> 3D Grid Image (C, H_grid, W_grid)
            # nrow=4로 설정하여 4개를 가로로 나열
            grid_tensor = make_grid(sample_images, nrow=4, padding=2)
            
            grid_image = wandb.Image(
                grid_tensor, 
                caption=f"Epoch {epoch+1}: Top(Original) / Bottom(Reconstructed)"
            )
            log_dict["Reconstruction_Vis"] = grid_image
            
        wandb.log(log_dict)
        
        # Scheduler Update
        scheduler.step(avg_val_loss)
        
        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_final.pth")
            wandb.save("best_model_final.pth") # WandB 클라우드에도 업로드
            print(f"⭐ New Best Model Saved! (Val Loss: {best_val_loss:.6f})")
            
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000 # 초 단위
        print(f"   -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {elapsed_time:.1f}s")

    # 6. 학습 종료 및 최종 모델 저장
    torch.save(model.state_dict(), "final_model_100ep.pth")
    wandb.save("final_model_100ep.pth")
    
    print("="*60)
    print(f"🎉 모든 학습 완료! 최종 Val Loss: {avg_val_loss:.6f}")
    print(f"🏆 Best Val Loss: {best_val_loss:.6f}")
    print("="*60)
    
    wandb.finish()

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # 필요한 폴더 생성 및 체크
    if not os.path.exists(CONFIG['data_dir']):
        print(f"❌ 경고: '{CONFIG['data_dir']}' 폴더가 없습니다. 경로를 확인해주세요.")
    else:
        train_final_model()