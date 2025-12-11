import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import librosa
import dlib
import joblib
import os
import glob
from tqdm import tqdm
from types import SimpleNamespace

# ============================================================
# 1. 사용자 설정 (경로 수정 필요)
# ============================================================
VIDEO_SOURCE_DIR = "./Deepfake_youtube_clip"  # ⚠️ 정상 영상들이 들어있는 폴더 경로
OUTPUT_CSV_PATH = "batch_validation_result.csv"

DLIB_PATH = "shape_predictor_68_face_landmarks.dat"
WHISPER_SIZE = "base"

# VAD 설정 (에너지 기반, 강력한 탐지)
VAD_SR = 22050
VAD_MIN_DURATION = 3.0

# 📊 학습 데이터 도메인 기준 (1920x1080 기준)
DOMAIN_CONFIG = {
    "target_width": 1920,
    "target_height": 1080,
    "target_area": 1920 * 1080,
    "target_sharpness": 68.67,
    "sharpness_std": 89.81
}

# 임계값 및 가중치
THRESHOLDS = {
    'tab': {'loose': 2.0, 'strict': 2.5, 'max': 3.5},
    'rnn': {'loose': 20.0, 'strict': 30.0, 'max': 50.0},
    'multi': {'loose': 40.0, 'strict': 60.0, 'max': 80.0}
}
WEIGHTS = {'xgb': 0.1, 'rnn': 0.4, 'multi': 0.5}

# 모델 파일 경로
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACIAL_LANDMARKS = {"left_eye": list(range(36, 42)), "right_eye": list(range(42, 48)), "nose": list(range(27, 36)), "mouth": list(range(48, 68)), "jawline": list(range(0, 17)), "full_face": list(range(0, 68))}

# ============================================================
# 2. 모델 클래스 정의 (webapp과 동일)
# ============================================================
class TabularAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Linear(128, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Linear(128, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

class RNNAE(nn.Module):
    def __init__(self, rnn_type, hidden_dim, num_layers, input_dim=5):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) if rnn_type == 'LSTM' else nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dec = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True) if rnn_type == 'LSTM' else nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.rnn_type = rnn_type
    def forward(self, x):
        if self.rnn_type == 'LSTM': _, (h, _) = self.enc(x)
        else: _, h = self.enc(x)
        h_rep = h[-1].unsqueeze(1).repeat(1, 90, 1)
        dec_out, _ = self.dec(h_rep)
        return self.out(dec_out)

class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cnn_encoder = nn.Sequential(nn.Conv2d(1, 32, 5, 2, 0), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 15 * 15, cfg.cnn_latent_dim), nn.ReLU())
        self.cnn_decoder = nn.Sequential(nn.Unflatten(1, (64, 16, 16)), nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(), nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid())
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
        if self.cfg.rnn_model == 'LSTM': _, (h_n, _) = self.rnn_encoder(npy)
        else: _, h_n = self.rnn_encoder(npy)
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        rnn_out, _ = self.rnn_decoder(self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, 90, 1))
        return self.cnn_decoder(self.cnn_decoder_fc(z)), self.rnn_output_layer(rnn_out)

# ============================================================
# 3. 헬퍼 함수
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
        if region_crop.size == 0: return None
        laplacian = cv2.Laplacian(region_crop, cv2.CV_64F)
        return {
            'laplacian_mean': np.abs(laplacian).mean(), 'laplacian_var': laplacian.var(),
            'light_intensity_mean': region_crop.mean(), 
            'light_intensity_change': (region_crop.mean() - prev_region_mean) if prev_region_mean else 0.0,
            'region_area': (x_max - x_min) * (y_max - y_min)
        }
    except: return None

def load_models():
    models = {}
    models['xgb'] = xgb.Booster(); models['xgb'].load_model(MODEL_PATHS['HQ']['xgb'])
    models['tab_ae'] = TabularAE(120, 64).to(device); models['tab_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['tab_ae'], map_location=device)); models['tab_ae'].eval()
    models['rnn_ae'] = RNNAE('GRU', 128, 2, 5).to(device); models['rnn_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['rnn_ae'], map_location=device)); models['rnn_ae'].eval()
    models['multi_model'] = MultiModalAutoencoder(SimpleNamespace(cnn_latent_dim=64, rnn_units=64, bottleneck_dim=64, rnn_model='LSTM')).to(device)
    models['multi_model'].load_state_dict(torch.load(MODEL_PATHS['HQ']['multi_ae'], map_location=device))
    models['multi_model'].eval()
    models['tab_scaler'] = joblib.load(MODEL_PATHS['HQ']['tab_scaler'])
    models['npy_scaler'] = joblib.load(MODEL_PATHS['HQ']['npy_scaler'])
    return models

def aggregate_tabular_features(frame_features):
    all_values = {region: {key: [] for key in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 'light_intensity_change', 'region_area']} for region in FACIAL_LANDMARKS.keys()}
    for frame_feat in frame_features:
        for region, feat in frame_feat.items():
            for key in feat.keys(): all_values[region][key].append(feat[key])
    aggregated = []
    for region in FACIAL_LANDMARKS.keys():
        for key in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 'light_intensity_change', 'region_area']:
            values = all_values[region][key]
            if values: aggregated.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])
            else: aggregated.extend([0.0]*4)
    return np.array(aggregated).reshape(1, -1)

def create_npy_features(frame_features, target_length=90):
    npy_data = []
    for frame_feat in frame_features:
        if frame_feat.get('full_face'):
            ff = frame_feat['full_face']
            npy_data.append([ff['laplacian_mean'], ff['laplacian_var'], ff['light_intensity_mean'], ff['light_intensity_change'], ff['region_area']])
    npy_array = np.array(npy_data)
    if len(npy_array) < target_length:
        npy_array = np.vstack([npy_array, np.zeros((target_length-len(npy_array), 5))])
    else: npy_array = npy_array[:target_length]
    return npy_array.reshape(1, target_length, 5)

def extract_audio_features(video_path, start_time, end_time):
    try:
        y, sr = librosa.load(video_path, sr=16000, duration=end_time-start_time, offset=start_time)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
        mel = cv2.resize(mel, (128,128))
        return ((mel - mel.min()) / (mel.max() - mel.min())).reshape(1,1,128,128)
    except: return None

# ============================================================
# 4. 핵심 로직: 배치 처리용
# ============================================================

def find_first_valid_segment(video_path):
    """
    [Batch용] 영상 전체에서 VAD로 목소리를 찾고, 
    그 중에서 얼굴이 검출되는 첫 번째 3초 구간을 반환
    """
    try:
        y, sr = librosa.load(video_path, sr=VAD_SR)
        
        # 에너지 기반 윈도우 슬라이딩 (강력한 VAD)
        frame_len, hop_len = 2048, 512
        rmse = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        
        # 3초 구간에 해당하는 프레임 수
        target_frames = int((VAD_MIN_DURATION * sr) / hop_len)
        
        if len(rmse) <= target_frames:
            # 영상이 너무 짧으면 처음부터 끝까지 (단, 1초 이상이어야 함)
            return (0.0, min(3.0, librosa.get_duration(y=y, sr=sr))) if len(y) > sr else None

        # 에너지가 높은 순서대로 상위 5개 후보 구간 추출
        # (단순히 제일 큰거 하나만 찾으면 얼굴이 없을 수도 있으므로)
        window = np.ones(target_frames)
        energy_sums = np.convolve(rmse, window, mode='valid')
        
        # 상위 5개 인덱스 (겹치지 않게 하려면 복잡해지므로, 여기선 단순하게 등간격 탐색으로 변경)
        # 성능을 위해: 0초, 5초, 10초... 지점에서 3초씩 얼굴 확인
        duration = librosa.get_duration(y=y, sr=sr)
        
        cap = cv2.VideoCapture(video_path)
        detector = dlib.get_frontal_face_detector()
        
        # 0초부터 5초 간격으로 탐색하며 얼굴이 있는지 확인
        for start_sec in range(0, int(duration - VAD_MIN_DURATION), 3):
            # 해당 구간의 에너지 확인 (너무 조용하면 패스)
            start_frame = int((start_sec * sr) / hop_len)
            end_frame = start_frame + target_frames
            if end_frame < len(rmse) and np.mean(rmse[start_frame:end_frame]) < 0.005: # Threshold for silence
                continue
                
            # 얼굴 확인 (중간 지점 프레임 하나만 확인)
            check_time = start_sec + 1.5
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(check_time * cap.get(cv2.CAP_PROP_FPS)))
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if len(detector(gray)) > 0:
                    cap.release()
                    return (float(start_sec), float(start_sec + VAD_MIN_DURATION))
        
        cap.release()
        return None # 적절한 구간 못 찾음
        
    except Exception as e:
        # print(f"Error in VAD: {e}")
        return None

def process_single_video(video_path, models, detector, predictor):
    """단일 비디오 분석 및 결과 반환"""
    
    # 1. 첫 번째 유효 구간(음성+얼굴) 탐색
    segment = find_first_valid_segment(video_path)
    if not segment:
        return None # 유효 구간 없음
    
    start_time, end_time = segment
    
    # 2. 특징 추출 (v13 로직: No Resizing)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    total_frames = int((end_time - start_time) * fps)
    
    frame_features = []
    prev_regions = {}
    sharpness_values = []
    processed = 0
    
    while processed < total_frames:
        ret, frame = cap.read()
        if not ret: break
        processed += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness_values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        faces = detector(gray)
        if len(faces) > 0:
            shape = predictor(gray, faces[0])
            frame_feat = {}
            for region_name, indices in FACIAL_LANDMARKS.items():
                prev_mean = prev_regions.get(region_name)
                region_feat = calculate_region_features(gray, shape, region_name, indices, prev_mean)
                if region_feat:
                    frame_feat[region_name] = region_feat
                    prev_regions[region_name] = region_feat['light_intensity_mean']
            if frame_feat: frame_features.append(frame_feat)
    cap.release()
    
    if not frame_features: return None
    
    input_sharpness = np.mean(sharpness_values) if sharpness_values else 0.0
    
    # 3. 예측
    tab_feat = aggregate_tabular_features(frame_features)
    npy_feat = create_npy_features(frame_features)
    aud_feat = extract_audio_features(video_path, start_time, end_time)
    
    tab_scaled = models['tab_scaler'].transform(tab_feat)
    tab_tensor = torch.FloatTensor(tab_scaled).to(device)
    npy_scaled = models['npy_scaler'].transform(npy_feat.reshape(-1, 5)).reshape(1, 90, 5)
    npy_tensor = torch.FloatTensor(npy_scaled).to(device)
    
    scores = {}
    scores['xgb'] = models['xgb'].predict(xgb.DMatrix(tab_scaled))[0]
    
    with torch.no_grad():
        tab_recon = models['tab_ae'](tab_tensor)
        # scores['tab'] = torch.mean((tab_tensor - tab_recon) ** 2).item() 
        # Tabular AE 점수는 현재 로직에서 크게 안쓰이므로 생략 가능하나 계산은 함
        
        rnn_recon = models['rnn_ae'](npy_tensor)
        raw_rnn = torch.mean((npy_tensor - rnn_recon) ** 2).item()
        
        if aud_feat is not None:
            aud_t = torch.FloatTensor(aud_feat).to(device)
            img_r, npy_r = models['multi_model'](aud_t, npy_tensor)
            raw_multi = (torch.mean((aud_t - img_r)**2) + torch.mean((npy_tensor - npy_r)**2)).item()
        else: raw_multi = 0.0
        
    # 4. 점수 보정 (Safety Clamp 적용)
    area_ratio = (width * height) / DOMAIN_CONFIG['target_area']
    area_factor = max(1.0, area_ratio)
    
    sharp_ratio = input_sharpness / DOMAIN_CONFIG['target_sharpness']
    sharp_factor = max(1.0, sharp_ratio)
    
    final_factor = area_factor * sharp_factor
    
    scores['rnn'] = raw_rnn / final_factor
    scores['multi'] = raw_multi / final_factor
    
    # 5. 결과 판정 (정상 영상 기준)
    # 임계값보다 낮으면 '정상(Real)' -> Success: T
    # 하나라도 임계값을 한참 넘으면 '딥페이크(Fake)' -> Success: F
    
    # 가중치 합산 점수 계산
    # 0: 정상, 0.3: 주의, 0.7: 의심, 1.0: 높음
    def get_prob(score, t):
        if score <= t['loose']: return 0.0
        elif score <= t['strict']: return 0.3
        elif score <= t['max']: return 0.7
        return 1.0
    
    p_xgb = scores['xgb'] # XGB는 확률값 그대로 사용
    p_rnn = get_prob(scores['rnn'], THRESHOLDS['rnn'])
    p_multi = get_prob(scores['multi'], THRESHOLDS['multi'])
    
    # 오디오가 없으면 Multi 제외
    if aud_feat is None:
        final_prob = (p_xgb * 0.4) + (p_rnn * 0.6)
    else:
        final_prob = (p_xgb * WEIGHTS['xgb']) + (p_rnn * WEIGHTS['rnn']) + (p_multi * WEIGHTS['multi'])
    
    final_prob_percent = final_prob * 100
    
    # 입력이 '정상 영상'이므로, 50% 미만이면 성공(T), 이상이면 실패(F)
    is_success = 'T' if final_prob_percent < 50 else 'F'
    
    return {
        'success': is_success,
        'XGBoost': round(float(scores['xgb']), 4),
        'RNN': round(scores['rnn'], 4),
        'Multi': round(scores['multi'], 4)
    }

# ============================================================
# 5. 메인 실행
# ============================================================
def main():
    # 결과 저장용 리스트
    results = []
    
    print(f"🚀 Batch Validation Start")
    print(f"📂 Source: {VIDEO_SOURCE_DIR}")
    
    # 모델 로드
    print("📦 Loading models...")
    try:
        models = load_models()
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 파일 목록 가져오기
    video_files = sorted(glob.glob(os.path.join(VIDEO_SOURCE_DIR, "*")))
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    
    print(f"🔍 Found {len(video_files)} videos.")
    
    # 루프 실행
    for idx, video_path in enumerate(tqdm(video_files)):
        file_name = os.path.basename(video_path)
        
        try:
            res = process_single_video(video_path, models, detector, predictor)
            
            if res:
                results.append({
                    'id_org': idx + 1,
                    'name': file_name,
                    'success': res['success'],
                    'XGBoost': res['XGBoost'],
                    'RNN': res['RNN'],
                    'Multi': res['Multi']
                })
            else:
                # 얼굴이나 음성을 못 찾은 경우 (Error 로깅)
                results.append({
                    'id_org': idx + 1,
                    'name': file_name,
                    'success': 'Skip',
                    'XGBoost': 0, 'RNN': 0, 'Multi': 0
                })
                
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            results.append({
                'id_org': idx + 1,
                'name': file_name,
                'success': 'Error',
                'XGBoost': 0, 'RNN': 0, 'Multi': 0
            })

    # CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ 모든 작업 완료. 결과 저장됨: {OUTPUT_CSV_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()