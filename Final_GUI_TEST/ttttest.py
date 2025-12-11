import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import librosa
import whisper
import dlib
import joblib
import os
import glob
from types import SimpleNamespace
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리 (pip install tqdm)

# ============================================================
# 1. 설정 및 상수
# ============================================================
DLIB_PATH = "shape_predictor_68_face_landmarks.dat"
WHISPER_SIZE = "base"

# 모델 경로 (기존 코드와 동일하게 설정)
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

THRESHOLDS = {
    'tab': {'loose': 0.03, 'strict': 0.05, 'max': 0.15},
    'rnn': {'loose': 7.0, 'strict': 10.0, 'max': 15.0},
    'multi': {'loose': 10.0, 'strict': 20.0, 'max': 30.0}
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
# 2. 모델 클래스 정의 (기존 코드 유지)
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
# 3. 특징 추출 및 로더 함수
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

def load_models():
    """모델 로드"""
    print("📥 모델을 메모리로 로드 중...")
    models = {}
    
    # XGBoost
    models['xgb'] = xgb.Booster()
    models['xgb'].load_model(MODEL_PATHS['HQ']['xgb'])
    
    # Tabular AE
    models['tab_ae'] = TabularAE(120, 64).to(device)
    models['tab_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['tab_ae'], map_location=device))
    models['tab_ae'].eval()
    
    # RNN AE
    models['rnn_ae'] = RNNAE('GRU', 128, 2, 5).to(device)
    models['rnn_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['rnn_ae'], map_location=device))
    models['rnn_ae'].eval()
    
    # Scalers
    models['tab_scaler'] = joblib.load(MODEL_PATHS['HQ']['tab_scaler'])
    models['npy_scaler'] = joblib.load(MODEL_PATHS['HQ']['npy_scaler'])
    
    print("✅ 모델 로드 완료")
    return models

def load_multimodal_model():
    cfg = SimpleNamespace(
        cnn_latent_dim=64,
        rnn_units=64,
        bottleneck_dim=64,
        rnn_model='LSTM'
    )
    model = MultiModalAutoencoder(cfg).to(device)
    model.load_state_dict(torch.load(MODEL_PATHS['HQ']['multi_ae'], map_location=device))
    model.eval()
    return model

def find_valid_segment(video_path, required_duration=3.0):
    """
    영상에서 '1명의 얼굴만 감지되는 연속된 3초 구간'을 탐색합니다.
    오디오가 있는지도 확인합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    # 필요한 연속 프레임 수 계산
    required_frames = int(required_duration * fps)
    
    detector = dlib.get_frontal_face_detector()
    
    frame_count = 0
    consecutive_faces = 0
    start_frame_idx = 0
    found_segment = False
    
    # 최적화를 위해 전체를 다 돌지 않고, 조건을 만족하면 즉시 반환
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 성능을 위해 프레임 크기 축소하여 얼굴 감지 (속도 향상)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # 얼굴이 정확히 1명인 경우
        if len(faces) == 1:
            if consecutive_faces == 0:
                start_frame_idx = frame_count - 1 # 시작점 기록
            consecutive_faces += 1
        else:
            consecutive_faces = 0 # 초기화
            
        # 3초 이상 연속되면 성공
        if consecutive_faces >= required_frames:
            found_segment = True
            break
            
    cap.release()
    
    if found_segment:
        start_time = start_frame_idx / fps
        end_time = (start_frame_idx + consecutive_faces) / fps
        # 정확히 3초만 자름
        end_time = start_time + required_duration
        return start_time, end_time
    
    return None

def extract_features(video_path, start_time, end_time, use_audio=True):
    """특정 구간에 대한 특징 추출"""
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PATH)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_features = []
    prev_regions = {}
    
    total_frames = end_frame - start_frame
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
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
    
    cap.release()
    
    if not frame_features:
        return None, None, None
    
    # 특징 집계
    tab_features = aggregate_tabular_features(frame_features)
    npy_features = create_npy_features(frame_features)
    
    # 음성 특징 (3초 구간)
    audio_features = None
    if use_audio:
        try:
            audio_features = extract_audio_features(video_path, start_time, end_time)
        except Exception as e:
            # 오디오 추출 실패 시 None 반환 (분석에서 제외)
            pass
            
    return tab_features, npy_features, audio_features

def aggregate_tabular_features(frame_features):
    all_values = {region: {key: [] for key in ['laplacian_mean', 'laplacian_var', 
                                                 'light_intensity_mean', 'light_intensity_change', 
                                                 'region_area']} 
                  for region in FACIAL_LANDMARKS.keys()}
    
    for frame_feat in frame_features:
        for region, feat in frame_feat.items():
            for key in feat.keys():
                all_values[region][key].append(feat[key])
    
    aggregated = []
    for region in FACIAL_LANDMARKS.keys():
        for key in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 
                    'light_intensity_change', 'region_area']:
            values = all_values[region][key]
            if values:
                aggregated.append(np.mean(values))
                aggregated.append(np.std(values))
                aggregated.append(np.min(values))
                aggregated.append(np.max(values))
            else:
                aggregated.extend([0.0, 0.0, 0.0, 0.0])
    
    return np.array(aggregated).reshape(1, -1)

def create_npy_features(frame_features, target_length=90):
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
    if len(npy_array) < target_length:
        pad = np.zeros((target_length - len(npy_array), 5))
        npy_array = np.vstack([npy_array, pad])
    else:
        npy_array = npy_array[:target_length]
    return npy_array.reshape(1, target_length, 5)

def extract_audio_features(video_path, start_time, end_time):
    # Whisper 로드 (이미 로드되어 있다면 재사용 가능하지만 여기선 함수 내 호출)
    model = whisper.load_model(WHISPER_SIZE)
    # Whisper는 전체 파일에 대해 동작하므로 try-except로 감쌈
    try:
        # 오디오 유무 확인용 로드
        y, sr = librosa.load(video_path, sr=16000, duration=end_time - start_time, offset=start_time)
        if len(y) == 0:
            return None
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_resized = cv2.resize(mel_spec_db, (128, 128))
        mel_normalized = (mel_resized - mel_resized.min()) / (mel_resized.max() - mel_resized.min())
        return mel_normalized.reshape(1, 1, 128, 128)
    except:
        return None

def predict_score(models, features):
    tab_features, npy_features, audio_features = features
    scores = {}
    
    # 1. Tabular
    tab_scaled = models['tab_scaler'].transform(tab_features)
    tab_tensor = torch.FloatTensor(tab_scaled).to(device)
    
    # 2. NPY (RNN)
    npy_scaled = models['npy_scaler'].transform(npy_features.reshape(-1, 5)).reshape(1, 90, 5)
    npy_tensor = torch.FloatTensor(npy_scaled).to(device)
    
    # XGBoost
    dmat = xgb.DMatrix(tab_scaled)
    scores['xgb'] = models['xgb'].predict(dmat)[0]
    
    # RNN AE
    with torch.no_grad():
        rnn_recon = models['rnn_ae'](npy_tensor)
        scores['rnn'] = torch.mean((npy_tensor - rnn_recon) ** 2).item()
    
    # Multi Model (Audio 있으면)
    if audio_features is not None:
        multi_model = load_multimodal_model()
        audio_tensor = torch.FloatTensor(audio_features).to(device)
        with torch.no_grad():
            img_recon, npy_recon = multi_model(audio_tensor, npy_tensor)
            scores['multi'] = (torch.mean((audio_tensor - img_recon) ** 2) + 
                             torch.mean((npy_tensor - npy_recon) ** 2)).item()
    else:
        scores['multi'] = -1.0 # 오디오 없음 표시
        
    return scores

def calculate_final_prob(scores):
    """점수를 확률로 변환"""
    th = THRESHOLDS
    
    # Threshold 로직 (기존과 동일)
    def get_prob(score, t_dict):
        if score <= t_dict['loose']: return 0.0
        elif score <= t_dict['strict']: return 0.3
        elif score <= t_dict['max']: return 0.7
        else: return 1.0

    p_xgb = scores['xgb']
    p_rnn = get_prob(scores['rnn'], th['rnn'])
    
    if scores['multi'] != -1.0:
        p_multi = get_prob(scores['multi'], th['multi'])
        final_score = (p_xgb * WEIGHTS['xgb']) + (p_rnn * WEIGHTS['rnn']) + (p_multi * WEIGHTS['multi'])
    else:
        # 오디오 없는 경우 가중치 재조정
        w_xgb, w_rnn = 0.4, 0.6
        final_score = (p_xgb * w_xgb) + (p_rnn * w_rnn)
        
    return final_score * 100

# ============================================================
# 4. 메인 실행 로직
# ============================================================
def main(folder_path):
    # 지원 확장자
    exts = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm']
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not video_files:
        print(f"❌ '{folder_path}' 폴더에 비디오 파일이 없습니다.")
        return

    print(f"📂 분석 대상: 총 {len(video_files)}개 영상")
    
    # 모델 로드
    try:
        models = load_models()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 통계용 변수
    stats = {
        "total": len(video_files),
        "processed": 0,
        "skipped": 0,
        "fake": 0,
        "real": 0,
        "errors": 0
    }
    
    results_detail = []

    print("\n🚀 딥페이크 분석 시작...\n")
    
    for video_path in tqdm(video_files, desc="Processing"):
        filename = os.path.basename(video_path)
        
        try:
            # 1. 유효 구간 탐색 (1인 얼굴 + 3초)
            segment = find_valid_segment(video_path, required_duration=3.0)
            
            if segment is None:
                stats['skipped'] += 1
                results_detail.append({'file': filename, 'result': 'SKIPPED', 'prob': 0, 'note': '조건 불충족(얼굴X or 짧음)'})
                continue
            
            start_t, end_t = segment
            
            # 2. 특징 추출
            features = extract_features(video_path, start_t, end_t, use_audio=True)
            if features[0] is None:
                stats['skipped'] += 1
                results_detail.append({'file': filename, 'result': 'SKIPPED', 'prob': 0, 'note': '특징 추출 실패'})
                continue
                
            # 3. 예측
            scores = predict_score(models, features)
            prob = calculate_final_prob(scores)
            
            # 4. 결과 집계
            stats['processed'] += 1
            if prob >= 50.0:
                stats['fake'] += 1
                verdict = "FAKE"
            else:
                stats['real'] += 1
                verdict = "REAL"
                
            results_detail.append({
                'file': filename, 
                'result': verdict, 
                'prob': round(prob, 2), 
                'note': f"구간: {start_t:.1f}~{end_t:.1f}s"
            })
            
        except Exception as e:
            stats['errors'] += 1
            print(f"\n⚠️ 에러 발생 ({filename}): {e}")
            results_detail.append({'file': filename, 'result': 'ERROR', 'prob': 0, 'note': str(e)})

    # ============================================================
    # 최종 결과 리포트
    # ============================================================
    print("\n" + "="*50)
    print("📊 [분석 결과 리포트]")
    print("="*50)
    print(f"총 영상 개수: {stats['total']}개")
    print(f"분석 완료   : {stats['processed']}개")
    print(f"분석 불가   : {stats['skipped']}개 (조건 불충족)")
    print(f"오류 발생   : {stats['errors']}개")
    print("-" * 50)
    
    if stats['processed'] > 0:
        fake_ratio = (stats['fake'] / stats['processed']) * 100
        real_ratio = (stats['real'] / stats['processed']) * 100
        
        print(f"🚨 딥페이크 의심: {stats['fake']}개 ({fake_ratio:.1f}%)")
        print(f"✅ 정상 영상    : {stats['real']}개 ({real_ratio:.1f}%)")
    else:
        print("분석된 영상이 없습니다.")
    
    print("-" * 50)
    print("📄 [상세 목록]")
    # 결과 출력 (상위 20개만 출력하거나 전체 출력)
    for res in results_detail:
        icon = "🚨" if res['result'] == 'FAKE' else "✅" if res['result'] == 'REAL' else "⚠️"
        print(f"{icon} [{res['result']}] {res['file']} : {res['prob']}% ({res['note']})")

if __name__ == "__main__":
    # 여기에 분석할 비디오들이 들어있는 폴더 경로를 입력하세요
    TARGET_FOLDER = "./Celeb-synthesis"  
    
    main(TARGET_FOLDER)