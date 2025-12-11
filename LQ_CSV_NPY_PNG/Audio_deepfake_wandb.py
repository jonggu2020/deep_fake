import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import wandb
from tqdm import tqdm
import time

# ============================================================
# 0. 전역 설정 (여기만 수정하세요!)
# ============================================================
DATA_DIR = "./3_audio_spectrograms"  # 멜 스펙트로그램 이미지 폴더 경로
IMAGE_SIZE = 128                      # 이미지 크기
SILENCE_THRESHOLD = 10                # 침묵 구간 필터링 (0=비활성화)

# ============================================================
# 1. Dataset Class
# ============================================================
class MelSpectrogramDataset(Dataset):
    """멜 스펙트로그램 이미지 데이터셋"""
    
    def __init__(self, data_dir, transform=None, silence_threshold=10):
        self.data_dir = data_dir
        self.transform = transform
        self.silence_threshold = silence_threshold
        
        self.image_paths = []
        for fname in os.listdir(data_dir):
            if fname.endswith('.png'):
                fpath = os.path.join(data_dir, fname)
                if not self._is_silence(fpath):
                    self.image_paths.append(fpath)
        
        print(f"✅ 총 {len(self.image_paths)}개 이미지 로드 완료")
    
    def _is_silence(self, img_path):
        """침묵 구간 감지"""
        img = cv2.imread(img_path)
        if img is None:
            return True
        mean_intensity = np.mean(img)
        return mean_intensity < self.silence_threshold
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# ============================================================
# 2. Multiple CNN Architectures
# ============================================================

class ResNetAutoencoder(nn.Module):
    """ResNet 기반 오토인코더"""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder: Pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove FC and AvgPool
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 128x128
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class EfficientNetAutoencoder(nn.Module):
    """EfficientNet 기반 오토인코더"""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder: Pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.encoder = nn.Sequential(
            efficientnet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1280 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (1280, 4, 4)),
            
            nn.ConvTranspose2d(1280, 512, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 128x128
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class VGGAutoencoder(nn.Module):
    """VGG16 기반 오토인코더"""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder: Pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        self.encoder = nn.Sequential(
            vgg.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class SimpleConvAutoencoder(nn.Module):
    """간단한 Conv 오토인코더 (베이스라인)"""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ============================================================
# 3. Model Factory
# ============================================================
def create_model(model_name, latent_dim):
    """모델 팩토리"""
    models_dict = {
        'resnet18': ResNetAutoencoder,
        'efficientnet_b0': EfficientNetAutoencoder,
        'vgg16': VGGAutoencoder,
        'simple_conv': SimpleConvAutoencoder
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models_dict[model_name](latent_dim=latent_dim)

# ============================================================
# 4. Training with WandB
# ============================================================
def train_with_wandb(config=None):
    """WandB를 사용한 학습"""
    
    with wandb.init(config=config):
        config = wandb.config
        
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 데이터 로드
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Augmentation
            transforms.ToTensor(),
        ])
        
        dataset = MelSpectrogramDataset(
            data_dir=config.data_dir,
            transform=transform,
            silence_threshold=config.silence_threshold
        )
        
        # Train/Val split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 모델 생성
        model = create_model(config.model_name, config.latent_dim)
        model.to(device)
        
        # WandB에 모델 구조 로깅
        wandb.watch(model, log='all', log_freq=100)
        
        # Loss & Optimizer
        criterion = nn.MSELoss()
        
        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 학습 루프
        best_val_loss = float('inf')
        
        for epoch in range(config.num_epochs):
            # ===== Training =====
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
            for batch in train_pbar:
                images = batch.to(device)
                
                # Forward
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # ===== Validation =====
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
                for batch in val_pbar:
                    images = batch.to(device)
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Scheduler step
            scheduler.step(avg_val_loss)
            
            # WandB 로깅
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Best model 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))
                wandb.log({'best_val_loss': best_val_loss})
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # 최종 모델 저장
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'final_model.pth'))
        
        # 최종 메트릭 로깅
        wandb.log({'final_val_loss': avg_val_loss})
        
        return best_val_loss

# ============================================================
# 5. Phase 1: Sweep Configuration (빠른 탐색)
# ============================================================
sweep_config_phase1 = {
    'method': 'bayes',  # 베이지안 최적화
    'metric': {
        'name': 'best_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model_name': {
            'values': ['resnet18', 'efficientnet_b0', 'vgg16', 'simple_conv']
        },
        'latent_dim': {
            'values': [64, 128, 256, 512]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'optimizer': {
            'values': ['adam', 'adamw']
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.00001,
            'max': 0.001
        },
        'num_epochs': {
            'value': 30  # Phase 1은 빠르게
        },
        'image_size': {
            'value': 128
        },
        'silence_threshold': {
            'value': 10
        },
        'data_dir': {
            'value': DATA_DIR  # 전역 설정 사용
        }
    }
}

# ============================================================
# 6. Phase 2: Refined Sweep Configuration (정밀 탐색)
# ============================================================
def create_phase2_sweep_config(best_model, best_latent_dim, best_lr_range, best_wd_range, data_dir='./mel_spectrograms'):
    """Phase 1 결과를 바탕으로 Phase 2 sweep config 생성"""
    
    sweep_config_phase2 = {
        'method': 'bayes',
        'metric': {
            'name': 'best_val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'model_name': {
                'value': best_model  # 최적 모델 고정
            },
            'latent_dim': {
                'values': [max(32, best_latent_dim - 32), best_latent_dim, best_latent_dim + 32]  # 좁은 범위
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': best_lr_range[0],
                'max': best_lr_range[1]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'optimizer': {
                'values': ['adam', 'adamw']
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': best_wd_range[0],
                'max': best_wd_range[1]
            },
            'num_epochs': {
                'value': 50  # Phase 2는 더 길게
            },
            'image_size': {
                'value': 128
            },
            'silence_threshold': {
                'value': 10
            },
            'data_dir': {
                'value': data_dir  # 데이터 경로 추가
            }
        }
    }
    
    return sweep_config_phase2

# ============================================================
# 7. Main Execution
# ============================================================
if __name__ == "__main__":
    
    print("="*60)
    print("🔧 설정")
    print("="*60)
    print(f"데이터 경로: {DATA_DIR}")
    print(f"이미지 크기: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"침묵 임계값: {SILENCE_THRESHOLD}")
    print("="*60)
    
    # 데이터 경로 확인
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ 오류: 데이터 경로를 찾을 수 없습니다: {DATA_DIR}")
        print("💡 코드 상단의 DATA_DIR 변수를 올바른 경로로 수정하세요!")
        exit(1)
    
    # WandB 로그인 (첫 실행 시 API 키 입력 필요)
    # wandb.login()
    
    print("\n" + "="*60)
    print("🚀 Phase 1: 빠른 탐색 (다양한 모델 + 넓은 하이퍼파라미터)")
    print("="*60)
    
    # Phase 1 Sweep 생성
    sweep_id_phase1 = wandb.sweep(
        sweep_config_phase1,
        project="audio-deepfake-detection-phase1"
    )
    
    # Phase 1 실행 (각 모델당 10회)
    wandb.agent(sweep_id_phase1, function=train_with_wandb, count=40)  # 4개 모델 * 10회
    
    print("\n" + "="*60)
    print("✅ Phase 1 완료!")
    print("📊 WandB에서 결과를 확인하고 최적 설정을 찾으세요")
    print("="*60)
    
    # Phase 1 결과 분석 (수동으로 확인 후 아래 값 설정)
    # WandB 웹에서 가장 좋은 결과를 확인한 후:
    """
    best_model = 'resnet18'  # 예시
    best_latent_dim = 128
    best_lr_range = [0.0001, 0.001]
    best_wd_range = [0.00001, 0.0001]
    
    print("\n" + "="*60)
    print("🎯 Phase 2: 정밀 탐색 (최적 모델 + 좁은 하이퍼파라미터)")
    print("="*60)
    
    # Phase 2 Sweep 생성
    sweep_config_phase2 = create_phase2_sweep_config(
        best_model, best_latent_dim, best_lr_range, best_wd_range
    )
    
    sweep_id_phase2 = wandb.sweep(
        sweep_config_phase2,
        project="audio-deepfake-detection-phase2"
    )
    
    # Phase 2 실행 (100회)
    wandb.agent(sweep_id_phase2, function=train_with_wandb, count=100)
    
    print("\n" + "="*60)
    print("🎉 Phase 2 완료!")
    print("📊 최종 결과를 WandB에서 확인하세요")
    print("="*60)
    """