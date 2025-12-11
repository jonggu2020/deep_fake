import sys
import os
import io
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import librosa
import librosa.display
import whisper
import dlib
import joblib
import matplotlib.pyplot as plt
import traceback
from cv2 import dnn_superres  # [핵심] 외부 초해상도(Super Resolution) 라이브러리
from types import SimpleNamespace
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, QProgressBar, 
                             QCheckBox, QTextEdit, QGroupBox, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ============================================================
# 1. 설정 및 상수 (Configuration)
# ============================================================
DLIB_PATH = "shape_predictor_68_face_landmarks.dat"
WHISPER_SIZE = "base"
EDSR_MODEL_PATH = "./models/EDSR_x4.pb"  # [필수] 다운로드 받은 SR 모델 경로

MODEL_PATHS = {
    'HQ': {
        'xgb': './models/HQ/xgb_model.json',         
        'tab_ae': './models/HQ/tabular_ae.pth', 
        'rnn_ae': './models/HQ/rnn_ae.pth', 
        'multi_ae': './models/HQ/best_multimodal_ae_torch_ram.pt', 
        'tab_scaler': './models/HQ/tab_scaler.joblib', 
        'npy_scaler': './models/HQ/npy_scaler.joblib' 
    }
}

# [중요] HQ 모델 기준 임계값
THRESHOLDS = {
    'tab': {'loose': 0.03, 'strict': 0.05, 'max': 0.15},
    'rnn': {'loose': 7.0, 'strict': 10.0, 'max': 15.0},
    'multi': {'loose': 10.0, 'strict': 20.0, 'max': 30.0}
}

# 모델 가중치
WEIGHTS = {'xgb': 0.1, 'rnn': 0.4, 'multi': 0.5}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACIAL_LANDMARKS = {
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "nose": list(range(27, 36)),
    "mouth": list(range(48, 68)),
    "jawline": list(range(0, 17)),
    "full_face": list(range(0, 68))
}

# ============================================================
# 2. 헬퍼 함수 (Feature Extraction Logic)
# ============================================================
def get_region_bounding_box(shape, landmark_indices):
    points = [(shape.part(i).x, shape.part(i).y) for i in landmark_indices]
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))

def calculate_region_features(gray_frame, shape, region_name, landmark_indices, prev_region_mean=None):
    try:
        x_min, y_min, x_max, y_max = get_region_bounding_box(shape, landmark_indices)
        h, w = gray_frame.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        region_crop = gray_frame[y_min:y_max, x_min:x_max]
        
        if region_crop.size == 0:
            return None
        
        laplacian = cv2.Laplacian(region_crop, cv2.CV_64F)
        laplacian_mean = np.abs(laplacian).mean()
        laplacian_var = laplacian.var()
        light_intensity_mean = region_crop.mean()
        
        light_intensity_change = 0.0
        if prev_region_mean is not None:
            light_intensity_change = light_intensity_mean - prev_region_mean
        
        region_area = (x_max - x_min) * (y_max - y_min)
        
        return {
            'laplacian_mean': laplacian_mean,
            'laplacian_var': laplacian_var,
            'light_intensity_mean': light_intensity_mean,
            'light_intensity_change': light_intensity_change,
            'region_area': region_area
        }
    except:
        return None

# ============================================================
# 3. 모델 클래스 정의
# ============================================================
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
    def __init__(self, rnn_type, hidden_dim, num_layers, input_dim=5):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) if rnn_type == 'LSTM' else nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dec = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True) if rnn_type == 'LSTM' else nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.rnn_type = rnn_type
        
    def forward(self, x):
        if self.rnn_type == 'LSTM':
             _, (h, _) = self.enc(x)
        else:
             _, h = self.enc(x)
        
        h_rep = h[-1].unsqueeze(1).repeat(1, 90, 1)
        dec_out, _ = self.dec(h_rep)
        return self.out(dec_out)

class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        self.cfg = cfg
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 0), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 15 * 15, cfg.cnn_latent_dim), nn.ReLU()
        )
        self.cnn_decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        if cfg.rnn_model == 'LSTM':
            self.rnn_encoder = nn.LSTM(5, cfg.rnn_units, batch_first=True)
            self.rnn_decoder = nn.LSTM(cfg.rnn_units, cfg.rnn_units, batch_first=True)
        else:
            self.rnn_encoder = nn.GRU(5, cfg.rnn_units, batch_first=True)
            self.rnn_decoder = nn.GRU(cfg.rnn_units, cfg.rnn_units, batch_first=True)

        self.bottleneck = nn.Sequential(nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim), nn.ReLU())
        self.rnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, cfg.rnn_units)
        self.rnn_output_layer = nn.Linear(cfg.rnn_units, 5)
        self.cnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, 64 * 16 * 16)

    def forward(self, img, npy):
        cnn_feat = self.cnn_encoder(img)
        if self.cfg.rnn_model == 'LSTM':
             _, (h_n, _) = self.rnn_encoder(npy)
        else:
             _, h_n = self.rnn_encoder(npy)
             
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        rnn_out, _ = self.rnn_decoder(self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, 90, 1))
        return self.cnn_decoder(self.cnn_decoder_fc(z)), self.rnn_output_layer(rnn_out)

# ============================================================
# 4. 분석 작업 스레드 (3가지 버전 비교 모드)
# ============================================================
class AnalysisThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, start_time, end_time, use_audio):
        super().__init__()
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.use_audio = use_audio
        self.sr_model = None

    def run(self):
        try:
            self.log_signal.emit("🚀 분석 시작 (HQ 모델)")
            self.log_signal.emit(f"⏱️ 분석 구간: {self.start_time}초 ~ {self.end_time}초")
            
            # 1. 영상 화질 확인
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            is_low_quality = (width < 1280 and height < 720)
            
            if is_low_quality:
                self.log_signal.emit(f"📊 저화질 영상 감지 ({width}x{height})")
                self.log_signal.emit("🔬 HQ 모델로 분석을 시작합니다...")
                
                # 원본 저화질만 분석
                results = {}
                
                self.log_signal.emit("\n" + "="*50)
                self.log_signal.emit("📹 원본 저화질 영상 분석 중...")
                self.log_signal.emit("="*50)
                results['original'] = self.analyze_video(preprocess_mode='none')
                results['denoised'] = None
                results['upscaled'] = None
                self.progress_signal.emit(100)
                
                self.result_signal.emit(results)
                
            else:
                # 고화질 영상은 기존 방식대로
                self.log_signal.emit(f"✅ 고화질 영상 ({width}x{height}) - 일반 분석 모드")
                results = {
                    'original': self.analyze_video(preprocess_mode='none'),
                    'denoised': None,
                    'upscaled': None
                }
                self.progress_signal.emit(100)
                self.result_signal.emit(results)
                
        except Exception as e:
            self.error_signal.emit(f"분석 오류: {str(e)}\n{traceback.format_exc()}")

    def init_sr_model(self):
        """EDSR Super Resolution 모델 초기화"""
        try:
            if not os.path.exists(EDSR_MODEL_PATH):
                self.log_signal.emit(f"   ⚠️ SR 모델 파일이 없습니다: {EDSR_MODEL_PATH}")
                return False
            
            self.sr_model = dnn_superres.DnnSuperResImpl_create()
            self.sr_model.readModel(EDSR_MODEL_PATH)
            self.sr_model.setModel("edsr", 4)
            return True
        except Exception as e:
            self.log_signal.emit(f"   ❌ SR 모델 로드 실패: {str(e)}")
            return False

    def apply_preprocessing(self, frame, mode):
        """프레임 전처리 적용"""
        if mode == 'none':
            return frame
        elif mode == 'denoise':
            # 노이즈 제거: Non-local Means Denoising
            return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        elif mode == 'upscale':
            if self.sr_model:
                return self.sr_model.upsample(frame)
            else:
                return frame
        return frame

    def analyze_video(self, preprocess_mode='none'):
        """단일 버전 영상 분석"""
        try:
            # 모델 로드
            self.log_signal.emit("📦 모델 로딩 중...")
            models = self.load_models()
            
            # 특징 추출
            self.log_signal.emit("🎬 영상 특징 추출 중...")
            features = self.extract_features(preprocess_mode)
            
            if features is None:
                return {'error': '특징 추출 실패'}
            
            # 예측
            self.log_signal.emit("🤖 딥페이크 분석 중...")
            scores = self.predict(models, features)
            
            # 결과 계산
            result = self.calculate_final_result(scores)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}

    def load_models(self):
        """모델 로드"""
        models = {}
        
        # XGBoost
        models['xgb'] = xgb.Booster()
        models['xgb'].load_model(MODEL_PATHS['HQ']['xgb'])
        
        # Tabular AE (120 features, 64 latent_dim)
        models['tab_ae'] = TabularAE(120, 64).to(device)
        models['tab_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['tab_ae'], map_location=device))
        models['tab_ae'].eval()
        
        # RNN AE (GRU, hidden_dim=128, num_layers=2)
        models['rnn_ae'] = RNNAE('GRU', 128, 2, 5).to(device)
        models['rnn_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['rnn_ae'], map_location=device))
        models['rnn_ae'].eval()
        
        # Multimodal AE (음성 사용 시)
        if self.use_audio:
            cfg = SimpleNamespace(
                cnn_latent_dim=64,      # 512 -> 64
                rnn_units=64,           # 256 -> 64
                bottleneck_dim=64,      # 128 -> 64
                rnn_model='LSTM'        # GRU -> LSTM (다시 수정)
            )
            models['multi_ae'] = MultiModalAutoencoder(cfg).to(device)
            models['multi_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['multi_ae'], map_location=device))
            models['multi_ae'].eval()
        
        # Scalers
        models['tab_scaler'] = joblib.load(MODEL_PATHS['HQ']['tab_scaler'])
        models['npy_scaler'] = joblib.load(MODEL_PATHS['HQ']['npy_scaler'])
        
        return models

    def extract_features(self, preprocess_mode):
        """특징 추출 (전처리 모드 적용)"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            start_frame = int(self.start_time * fps)
            end_frame = int(self.end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Face detector
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(DLIB_PATH)
            
            frame_features = []
            prev_regions = {}
            
            frame_idx = start_frame
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 전처리 적용
                frame = self.apply_preprocessing(frame, preprocess_mode)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                if len(faces) > 0:
                    face = faces[0]
                    shape = predictor(gray, face)
                    
                    frame_feat = {}
                    for region_name, indices in FACIAL_LANDMARKS.items():
                        prev_mean = prev_regions.get(region_name)
                        region_feat = calculate_region_features(gray, shape, region_name, indices, prev_mean)
                        
                        if region_feat:
                            frame_feat[region_name] = region_feat
                            prev_regions[region_name] = region_feat['light_intensity_mean']
                    
                    if frame_feat:
                        frame_features.append(frame_feat)
                
                frame_idx += 1
            
            cap.release()
            
            if not frame_features:
                return None
            
            # Tabular 특징
            tab_features = self.aggregate_tabular_features(frame_features)
            
            # NPY 특징 (시계열)
            npy_features = self.create_npy_features(frame_features)
            
            # 음성 특징 (사용 시)
            audio_features = None
            if self.use_audio:
                audio_features = self.extract_audio_features()
            
            return {
                'tabular': tab_features,
                'npy': npy_features,
                'audio': audio_features
            }
            
        except Exception as e:
            self.log_signal.emit(f"❌ 특징 추출 오류: {str(e)}")
            return None

    def aggregate_tabular_features(self, frame_features):
        """프레임별 특징을 통합하여 tabular 형태로 변환 (120개 특징)"""
        all_values = {region: {key: [] for key in ['laplacian_mean', 'laplacian_var', 
                                                     'light_intensity_mean', 'light_intensity_change', 
                                                     'region_area']} 
                      for region in FACIAL_LANDMARKS.keys()}
        
        for frame_feat in frame_features:
            for region, feat in frame_feat.items():
                for key in feat.keys():
                    all_values[region][key].append(feat[key])
        
        aggregated = []
        # 각 region, 각 feature에 대해 mean, std, min, max 계산 (5 features * 4 stats = 20 per region)
        for region in FACIAL_LANDMARKS.keys():
            for key in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 
                        'light_intensity_change', 'region_area']:
                values = all_values[region][key]
                if values:
                    aggregated.append(np.mean(values))   # mean
                    aggregated.append(np.std(values))    # std
                    aggregated.append(np.min(values))    # min
                    aggregated.append(np.max(values))    # max
                else:
                    aggregated.extend([0.0, 0.0, 0.0, 0.0])
        
        # 6 regions * 5 features * 4 stats = 120 features
        return np.array(aggregated).reshape(1, -1)

    def create_npy_features(self, frame_features, target_length=90):
        """시계열 특징 생성"""
        npy_data = []
        
        for frame_feat in frame_features:
            full_face = frame_feat.get('full_face')
            if full_face:
                npy_data.append([
                    full_face['laplacian_mean'],
                    full_face['laplacian_var'],
                    full_face['light_intensity_mean'],
                    full_face['light_intensity_change'],
                    full_face['region_area']
                ])
        
        npy_array = np.array(npy_data)
        
        # 길이 조정
        if len(npy_array) < target_length:
            pad = np.zeros((target_length - len(npy_array), 5))
            npy_array = np.vstack([npy_array, pad])
        else:
            npy_array = npy_array[:target_length]
        
        return npy_array.reshape(1, target_length, 5)

    def extract_audio_features(self):
        """음성 특징 추출"""
        try:
            # Whisper로 음성 텍스트 변환
            model = whisper.load_model(WHISPER_SIZE)
            result = model.transcribe(self.video_path)
            
            # 멜 스펙트로그램 생성
            y, sr = librosa.load(self.video_path, sr=16000, 
                                duration=self.end_time - self.start_time, 
                                offset=self.start_time)
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 128x128로 리사이즈
            mel_resized = cv2.resize(mel_spec_db, (128, 128))
            mel_normalized = (mel_resized - mel_resized.min()) / (mel_resized.max() - mel_resized.min())
            
            return mel_normalized.reshape(1, 1, 128, 128)
            
        except Exception as e:
            self.log_signal.emit(f"⚠️ 음성 특징 추출 실패: {str(e)}")
            return None

    def predict(self, models, features):
        """모델 예측"""
        scores = {}
        
        # Tabular 정규화
        tab_scaled = models['tab_scaler'].transform(features['tabular'])
        tab_tensor = torch.FloatTensor(tab_scaled).to(device)
        
        # NPY 정규화
        npy_scaled = models['npy_scaler'].transform(features['npy'].reshape(-1, 5)).reshape(1, 90, 5)
        npy_tensor = torch.FloatTensor(npy_scaled).to(device)
        
        # XGBoost 예측
        dmat = xgb.DMatrix(tab_scaled)
        scores['xgb'] = models['xgb'].predict(dmat)[0]
        
        # Tabular AE 예측
        with torch.no_grad():
            tab_recon = models['tab_ae'](tab_tensor)
            scores['tab'] = torch.mean((tab_tensor - tab_recon) ** 2).item()
        
        # RNN AE 예측
        with torch.no_grad():
            rnn_recon = models['rnn_ae'](npy_tensor)
            scores['rnn'] = torch.mean((npy_tensor - rnn_recon) ** 2).item()
        
        # Multimodal AE 예측 (음성 사용 시)
        if self.use_audio and features['audio'] is not None:
            audio_tensor = torch.FloatTensor(features['audio']).to(device)
            with torch.no_grad():
                img_recon, npy_recon = models['multi_ae'](audio_tensor, npy_tensor)
                scores['multi'] = (torch.mean((audio_tensor - img_recon) ** 2) + 
                                 torch.mean((npy_tensor - npy_recon) ** 2)).item()
        else:
            scores['multi'] = 0.0
        
        return scores

    def calculate_final_result(self, scores):
        """최종 결과 계산"""
        report_lines = []
        
        def analyze_threshold(score, th_dict, name):
            if score <= th_dict['loose']:
                prob = 0.0
                status = "✅ 정상"
            elif score <= th_dict['strict']:
                prob = 0.3
                status = "⚠️ 주의"
            elif score <= th_dict['max']:
                prob = 0.7
                status = "🔴 의심"
            else:
                prob = 1.0
                status = "🚨 높음"
            
            report_lines.append(f"[{name}] Score: {score:.4f} → {status}")
            return prob
        
        th = THRESHOLDS
        
        p_xgb = scores['xgb']
        report_lines.append(f"[XGBoost] Prob: {p_xgb:.4f}")

        p_rnn = analyze_threshold(scores['rnn'], th['rnn'], "RNN Model")
        
        if self.use_audio and scores['multi'] > 0:
            p_multi = analyze_threshold(scores['multi'], th['multi'], "Multi Model")
            final_score = (p_xgb * WEIGHTS['xgb']) + (p_rnn * WEIGHTS['rnn']) + (p_multi * WEIGHTS['multi'])
        else:
            p_multi = 0.0
            report_lines.append("[Multi Model] OFF (분석 제외)")
            w_xgb, w_rnn = 0.4, 0.6
            final_score = (p_xgb * w_xgb) + (p_rnn * w_rnn)

        details_str = "\n".join(report_lines)

        return {
            'final_prob': final_score * 100,
            'details': details_str,
            'raw_scores': scores
        }

# ============================================================
# 5. 메인 윈도우 (GUI) - 3가지 결과 표시
# ============================================================
class DeepfakeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Detector Pro (HQ Model)")
        self.setGeometry(100, 100, 600, 700)
        self.initUI()
        self.video_path = None
        self.duration = 0

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        lbl_title = QLabel("딥페이크 탐지 시스템 (HQ 모델)")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(lbl_title)

        file_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("검사할 비디오 파일을 선택하세요...")
        btn_browse = QPushButton("파일 열기")
        btn_browse.clicked.connect(self.open_file)
        file_layout.addWidget(self.path_input)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)
        
        info_group = QGroupBox("영상 정보 및 품질 자동 감지")
        info_layout = QVBoxLayout()
        self.lbl_info = QLabel("파일을 불러와주세요.")
        self.lbl_quality = QLabel("화질: -")
        self.lbl_quality.setFont(QFont("Arial", 12, QFont.Bold))
        self.lbl_quality.setStyleSheet("color: gray")
        info_layout.addWidget(self.lbl_info)
        info_layout.addWidget(self.lbl_quality)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        opt_group = QGroupBox("분석 옵션")
        opt_layout = QHBoxLayout()
        
        opt_layout.addWidget(QLabel("시작(초):"))
        self.txt_start = QLineEdit("0")
        opt_layout.addWidget(self.txt_start)
        
        opt_layout.addWidget(QLabel("종료(초):"))
        self.txt_end = QLineEdit("0")
        opt_layout.addWidget(self.txt_end)
        
        self.chk_audio = QCheckBox("음성 포함 정밀 검사 (Multimodal)")
        self.chk_audio.setChecked(True)
        opt_layout.addWidget(self.chk_audio)
        
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        self.btn_run = QPushButton("딥페이크 분석 시작")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_run.setStyleSheet("background-color: #007BFF; color: white; border-radius: 5px;")
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(150)
        layout.addWidget(self.log_area)
        
        # 결과 표시 영역
        self.lbl_result = QLabel("결과 대기 중")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_result.setStyleSheet("border: 2px solid gray; padding: 15px; background-color: #f0f0f0;")
        layout.addWidget(self.lbl_result)
        
        # 상세 결과 표시
        self.result_detail = QTextEdit()
        self.result_detail.setReadOnly(True)
        self.result_detail.setFixedHeight(200)
        self.result_detail.setStyleSheet("background-color: #f9f9f9; padding: 10px; font-size: 12px;")
        layout.addWidget(self.result_detail)
        
        central_widget.setLayout(layout)

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, '비디오 선택', '', 'Video Files (*.mp4 *.avi *.mkv *.mov *.webm)')
        if fname:
            self.video_path = fname
            self.path_input.setText(fname)
            self.check_video_info()

    def check_video_info(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log("❌ 영상을 열 수 없습니다.")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # 화질 표시 로직
        if width >= 1280 or height >= 720:
            q_text = "HIGH QUALITY (고화질)"
            color = "green"
        else:
            q_text = "LOW QUALITY (저화질)"
            color = "orange"
            
        self.lbl_info.setText(f"해상도: {width}x{height} | FPS: {fps:.1f} | 길이: {self.duration:.2f}초")
        self.lbl_quality.setText(f"자동 분류: {q_text}")
        self.lbl_quality.setStyleSheet(f"color: {color}")
        
        self.txt_end.setText(f"{self.duration:.1f}")
        self.log(f"영상 로드 완료: {q_text}")

    def log(self, msg):
        self.log_area.append(msg)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def run_analysis(self):
        if not self.video_path:
            QMessageBox.warning(self, "경고", "비디오 파일을 선택해주세요.")
            return
            
        try:
            start = float(self.txt_start.text())
            end = float(self.txt_end.text())
            if start < 0 or end > self.duration or start >= end:
                raise ValueError
        except:
            QMessageBox.warning(self, "경고", "시작/종료 시간이 올바르지 않습니다.")
            return

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_result.setText("분석 중...")
        self.lbl_result.setStyleSheet("background-color: #f0f0f0; border: 2px solid gray;")
        self.log_area.clear()
        self.result_detail.clear()
        
        # AnalysisThread 실행
        self.worker = AnalysisThread(self.video_path, start, end, self.chk_audio.isChecked())
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.result_signal.connect(self.show_result)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.start()

    def show_result(self, results):
        """원본 결과 표시"""
        
        # 원본 결과만 표시
        if results.get('original'):
            orig = results['original']
            if 'error' not in orig:
                prob = orig['final_prob']
                details = orig['details']
                
                if prob >= 50:
                    verdict = "⚠️ 딥페이크 의심"
                    style = "background-color: #FFDDDD; color: red; border: 2px solid red;"
                else:
                    verdict = "✅ 정상 영상 가능성 높음"
                    style = "background-color: #DDFFDD; color: green; border: 2px solid green;"
                
                self.lbl_result.setText(f"{verdict}\n확률: {prob:.2f}%")
                self.lbl_result.setStyleSheet(style)
                
                # 상세 결과
                detail_text = f"[상세 점수]\n\n{details}"
                self.result_detail.setText(detail_text)
                
                report_msg = f"딥페이크 확률: {prob:.2f}%\n\n[상세 리포트]\n{details}"
                self.log(f"\n{report_msg}")
                QMessageBox.information(self, "분석 완료", report_msg)
            else:
                self.lbl_result.setText(f"❌ 오류 발생")
                self.lbl_result.setStyleSheet("background-color: #FFE0E0; border: 2px solid red;")
                self.result_detail.setText(f"오류: {orig['error']}")
        
        self.btn_run.setEnabled(True)
        self.log("\n✅ 분석 완료!")

    def handle_error(self, msg):
        self.log(f"❌ {msg}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "오류 발생", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepfakeApp()
    window.show()
    sys.exit(app.exec_())