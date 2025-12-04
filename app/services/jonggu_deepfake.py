"""종구님 딥페이크 탐지 모델 서비스

Jonggu's deepfake detection model integration
- XGBoost, RNN AE, MultiModal AE 앙상블
- 얼굴 랜드마크 기반 특징 추출
- 음성 분석 (Whisper + librosa)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import librosa
import dlib
import joblib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Any

# ============================================================
# 모델 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models_jonggu"

DLIB_PATH = str(MODEL_DIR / "shape_predictor_68_face_landmarks.dat")
WHISPER_SIZE = "base"

VAD_SR = 22050
VAD_TOP_DB = 60
VAD_MIN_DURATION = 2.0

DOMAIN_CONFIG = {
    "target_width": 1920,
    "target_height": 1080,
    "target_area": 1920 * 1080,
    "target_sharpness": 68.67,
    "sharpness_std": 89.81
}

MODEL_PATHS = {
    'HQ': {
        'xgb': str(MODEL_DIR / "models/HQ/xgb_model.json"),
        'tab_ae': str(MODEL_DIR / "models/HQ/tabular_ae.pth"),
        'rnn_ae': str(MODEL_DIR / "models/HQ/rnn_ae.pth"),
        'multi_ae': str(MODEL_DIR / "models/HQ/best_multimodal_ae_torch_ram.pt"),
        'tab_scaler': str(MODEL_DIR / "models/HQ/tab_scaler.joblib"),
        'npy_scaler': str(MODEL_DIR / "models/HQ/npy_scaler.joblib")
    }
}

THRESHOLDS = {
    'tab': {'loose': 0.0215, 'strict': 0.0266, 'max': 0.1064},
    'rnn': {'loose': 0.5390, 'strict': 0.7137, 'max': 2.6497},
    'multi': {'loose': 0.6564, 'strict': 0.8600, 'max': 2.0305}
}

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
# 모델 클래스 정의
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

    def forward(self, x):
        return self.decoder(self.encoder(x))


class RNNAE(nn.Module):
    def __init__(self, rnn_type, hidden_dim, num_layers, input_dim=5):
        super().__init__()
        if rnn_type == 'LSTM':
            self.enc = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.dec = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.enc = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.dec = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
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

        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim),
            nn.ReLU()
        )
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
# 헬퍼 함수
# ============================================================

def get_region_bounding_box(shape, landmark_indices):
    """얼굴 영역의 바운딩 박스 계산"""
    points = [(shape.part(i).x, shape.part(i).y) for i in landmark_indices]
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))


def calculate_region_features(gray_frame, shape, region_name, landmark_indices, prev_region_mean=None):
    """얼굴 영역별 특징 추출"""
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


def detect_speech_segment(video_path: str, search_start: float, search_end: float) -> Tuple[Optional[Tuple[float, float]], str]:
    """음성 구간 감지"""
    try:
        duration = search_end - search_start
        if duration < VAD_MIN_DURATION:
            return None, "⚠️ 검색 범위가 너무 짧습니다."

        y, sr = librosa.load(video_path, sr=VAD_SR, offset=search_start, duration=duration)
        speech_intervals_samples = librosa.effects.split(y, top_db=VAD_TOP_DB, hop_length=512)
        speech_intervals_time = librosa.samples_to_time(speech_intervals_samples, sr=sr)

        valid_segments = []
        for start, end in speech_intervals_time:
            if end - start >= VAD_MIN_DURATION:
                valid_segments.append((start + search_start, end + search_start))

        if not valid_segments:
            return None, "⚠️ 지정된 구간 내에서 뚜렷한 목소리를 찾지 못했습니다."

        best_start = valid_segments[0][0]
        best_end = min(valid_segments[0][1], best_start + 5.0)
        return (best_start, best_end), f"✅ 분석 구간 확정: {best_start:.2f}초 ~ {best_end:.2f}초"
    except Exception as e:
        return None, f"❌ 오디오 분석 실패: {str(e)}"


def load_models():
    """모든 모델 로드"""
    models = {}
    try:
        models['xgb'] = xgb.Booster()
        models['xgb'].load_model(MODEL_PATHS['HQ']['xgb'])
        
        models['tab_ae'] = TabularAE(120, 64).to(device)
        models['tab_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['tab_ae'], map_location=device))
        models['tab_ae'].eval()
        
        models['rnn_ae'] = RNNAE('GRU', 128, 2, 5).to(device)
        models['rnn_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['rnn_ae'], map_location=device))
        models['rnn_ae'].eval()
        
        models['tab_scaler'] = joblib.load(MODEL_PATHS['HQ']['tab_scaler'])
        models['npy_scaler'] = joblib.load(MODEL_PATHS['HQ']['npy_scaler'])
        
        print("✅ [jonggu_deepfake] 모든 모델 로드 성공", file=sys.stderr, flush=True)
        return models
    except Exception as e:
        print(f"❌ [jonggu_deepfake] 모델 로드 실패: {e}", file=sys.stderr, flush=True)
        return None


def load_multimodal_model():
    """MultiModal 모델 로드"""
    try:
        cfg = SimpleNamespace(cnn_latent_dim=64, rnn_units=64, bottleneck_dim=64, rnn_model='LSTM')
        model = MultiModalAutoencoder(cfg).to(device)
        model.load_state_dict(torch.load(MODEL_PATHS['HQ']['multi_ae'], map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"❌ [jonggu_deepfake] MultiModal 모델 로드 실패: {e}", file=sys.stderr, flush=True)
        return None


def extract_features(video_path: str, start_time: float, end_time: float, use_audio: bool = False):
    """비디오에서 특징 추출"""
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_PATH)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_features = []
        prev_regions = {}
        sharpness_values = []

        total_frames = end_frame - start_frame
        processed_count = 0

        while processed_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            processed_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_values.append(curr_sharpness)

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

        cap.release()

        input_stats = {
            "mean": np.mean(sharpness_values) if sharpness_values else 0.0,
            "std": np.std(sharpness_values) if sharpness_values else 0.0,
            "min": np.min(sharpness_values) if sharpness_values else 0.0,
            "max": np.max(sharpness_values) if sharpness_values else 0.0
        }

        if not frame_features:
            return None, None, None, input_stats

        tab_features = aggregate_tabular_features(frame_features)
        npy_features = create_npy_features(frame_features)
        audio_features = None

        if use_audio:
            try:
                y, sr = librosa.load(video_path, sr=16000, duration=end_time-start_time, offset=start_time)
                mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
                mel = cv2.resize(mel, (128, 128))
                audio_features = ((mel - mel.min()) / (mel.max() - mel.min())).reshape(1, 1, 128, 128)
            except:
                audio_features = None

        return tab_features, npy_features, audio_features, input_stats
    except Exception as e:
        print(f"❌ [jonggu_deepfake] 특징 추출 실패: {e}", file=sys.stderr, flush=True)
        return None, None, None, None


def aggregate_tabular_features(frame_features):
    """Tabular 특징 집계"""
    all_values = {region: {key: [] for key in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 'light_intensity_change', 'region_area']}
                  for region in FACIAL_LANDMARKS.keys()}
    
    for frame_feat in frame_features:
        for region, feat in frame_feat.items():
            for key in feat.keys():
                all_values[region][key].append(feat[key])

    aggregated = []
    for region in FACIAL_LANDMARKS.keys():
        for key in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 'light_intensity_change', 'region_area']:
            values = all_values[region][key]
            if values:
                aggregated.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])
            else:
                aggregated.extend([0.0] * 4)

    return np.array(aggregated).reshape(1, -1)


def create_npy_features(frame_features, target_length=90):
    """NPY 특징 생성"""
    npy_data = []
    for frame_feat in frame_features:
        if frame_feat.get('full_face'):
            ff = frame_feat['full_face']
            npy_data.append([ff['laplacian_mean'], ff['laplacian_var'], 
                           ff['light_intensity_mean'], ff['light_intensity_change'], ff['region_area']])

    npy_array = np.array(npy_data)
    if len(npy_array) < target_length:
        npy_array = np.vstack([npy_array, np.zeros((target_length - len(npy_array), 5))])
    else:
        npy_array = npy_array[:target_length]

    return npy_array.reshape(1, target_length, 5)


def predict(models, features, use_audio: bool = False) -> Dict[str, float]:
    """딥페이크 확률 예측"""
    tab_features, npy_features, audio_features = features
    scores = {}

    tab_scaled = models['tab_scaler'].transform(tab_features)
    tab_tensor = torch.FloatTensor(tab_scaled).to(device)
    npy_scaled = models['npy_scaler'].transform(npy_features.reshape(-1, 5)).reshape(1, 90, 5)
    npy_tensor = torch.FloatTensor(npy_scaled).to(device)

    scores['xgb'] = models['xgb'].predict(xgb.DMatrix(tab_scaled))[0]

    with torch.no_grad():
        tab_recon = models['tab_ae'](tab_tensor)
        scores['tab'] = torch.mean((tab_tensor - tab_recon) ** 2).item()
        rnn_recon = models['rnn_ae'](npy_tensor)
        scores['rnn'] = torch.mean((npy_tensor - rnn_recon) ** 2).item()

    if use_audio and audio_features is not None:
        multi_model = load_multimodal_model()
        if multi_model:
            aud_t = torch.FloatTensor(audio_features).to(device)
            with torch.no_grad():
                img_r, npy_r = multi_model(aud_t, npy_tensor)
                scores['multi'] = (torch.mean((aud_t - img_r)**2) + torch.mean((npy_tensor - npy_r)**2)).item()
        else:
            scores['multi'] = 0.0
    else:
        scores['multi'] = 0.0

    return scores


def calculate_result(scores: Dict[str, float], input_stats: Dict[str, float], sensitivity_k: float = 2.0, use_audio: bool = False) -> Dict[str, Any]:
    """최종 결과 계산"""
    th = THRESHOLDS

    # 보정 계수
    sharp_factor = input_stats['mean'] / sensitivity_k
    final_factor = max(1.0, sharp_factor)

    # 점수 보정
    adj_scores = {
        'xgb': scores['xgb'],
        'rnn': scores['rnn'] / final_factor,
        'multi': scores['multi'] / final_factor
    }

    def get_status(score, t):
        if score <= t['loose']:
            return 0.0, "✅ 정상"
        elif score <= t['strict']:
            return 0.3, "⚠️ 주의"
        elif score <= t['max']:
            return 0.7, "🔴 의심"
        return 1.0, "🚨 높음"

    p_rnn, s_rnn = get_status(adj_scores['rnn'], th['rnn'])

    if use_audio and scores['multi'] > 0:
        p_multi, s_multi = get_status(adj_scores['multi'], th['multi'])
        final = (adj_scores['xgb'] * WEIGHTS['xgb']) + (p_rnn * WEIGHTS['rnn']) + (p_multi * WEIGHTS['multi'])
    else:
        final = (adj_scores['xgb'] * 0.4) + (p_rnn * 0.6)

    return {
        'fake_probability': final * 100,
        'is_fake': final >= 0.5,
        'scores': adj_scores,
        'input_sharpness': input_stats['mean'],
        'sensitivity_factor': final_factor
    }


async def detect_deepfake_from_file(video_file_path: str, sensitivity_k: float = 2.0, use_audio: bool = True) -> Dict[str, Any]:
    """
    비디오 파일에서 딥페이크 탐지
    
    Args:
        video_file_path: 비디오 파일 경로
        sensitivity_k: 민감도 상수 (기본값 2.0)
        use_audio: 음성 분석 포함 여부
    
    Returns:
        탐지 결과 딕셔너리
    """
    try:
        # 1. 비디오 메타데이터 추출
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            return {"error": "비디오 파일을 열 수 없습니다"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration < 3.0:
            return {"error": "비디오가 너무 짧습니다 (최소 3초)"}

        # 2. 음성 구간 감지
        search_end = min(duration, 15.0)
        vad_result, vad_msg = detect_speech_segment(video_file_path, 0.0, search_end)

        if vad_result is None:
            return {"error": vad_msg}

        start_time, end_time = vad_result

        # 3. 모델 로드
        models = load_models()
        if models is None:
            return {"error": "모델 로드 실패"}

        # 4. 특징 추출
        tab_features, npy_features, audio_features, input_stats = extract_features(
            video_file_path, start_time, end_time, use_audio
        )

        if tab_features is None:
            return {"error": "얼굴을 찾을 수 없습니다"}

        # 5. 예측
        scores = predict(models, (tab_features, npy_features, audio_features), use_audio)

        # 6. 결과 계산
        result = calculate_result(scores, input_stats, sensitivity_k, use_audio)

        return {
            "success": True,
            "fake_probability": result['fake_probability'],
            "is_fake": result['is_fake'],
            "analysis_range": {"start": start_time, "end": end_time},
            "input_sharpness": result['input_sharpness'],
            "sensitivity_factor": result['sensitivity_factor'],
            "scores": result['scores']
        }

    except Exception as e:
        print(f"❌ [jonggu_deepfake] 탐지 실패: {e}", file=sys.stderr, flush=True)
        return {"error": f"탐지 중 오류 발생: {str(e)}"}
