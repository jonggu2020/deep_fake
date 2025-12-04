import streamlit as st
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
import tempfile
import os
import time
from types import SimpleNamespace
import requests
import json

# ============================================================
# 1. 설정 및 상수
# ============================================================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dlib 파일 경로 (한글 경로 문제 회피를 위해 영문 경로 사용)
DLIB_PATH_ORIGINAL = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
DLIB_PATH_TEMP = r"C:\temp_dlib\shape_predictor_68_face_landmarks.dat"

# 영문 경로에 파일이 있으면 사용, 없으면 원본 경로 사용
if os.path.exists(DLIB_PATH_TEMP):
    DLIB_PATH = DLIB_PATH_TEMP
    print(f"✅ 영문 경로의 dlib 파일 사용: {DLIB_PATH}")
else:
    DLIB_PATH = DLIB_PATH_ORIGINAL
    print(f"⚠️  원본 경로의 dlib 파일 사용 (한글 경로 문제 발생 가능): {DLIB_PATH}")

WHISPER_SIZE = "base"

# 디버깅: 경로 출력
print(f"🔍 BASE_DIR: {BASE_DIR}")
print(f"🔍 DLIB_PATH: {DLIB_PATH}")
print(f"🔍 DLIB 파일 존재: {os.path.exists(DLIB_PATH)}")

VAD_SR = 22050
VAD_TOP_DB = 60 
VAD_MIN_DURATION = 2.0 

# ============================================================
# 백엔드 API 설정 및 세션 초기화
# ============================================================
BACKEND_API_URL = "http://localhost:8000"

# Streamlit 세션 상태 초기화
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "email" not in st.session_state:
    st.session_state.email = None
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

# 📊 [기본 도메인 설정] (사용자 제공: 1920x1080 기준)
DOMAIN_CONFIG = {
    "target_width": 1920,
    "target_height": 1080,
    "target_area": 1920 * 1080,
    "target_sharpness": 68.67,
    "sharpness_std": 89.81
}

MODEL_PATHS = {
    'HQ': {
        'xgb': os.path.join(BASE_DIR, 'models/HQ/xgb_model.json'),         
        'tab_ae': os.path.join(BASE_DIR, 'models/HQ/tabular_ae.pth'), 
        'rnn_ae': os.path.join(BASE_DIR, 'models/HQ/rnn_ae.pth'), 
        'multi_ae': os.path.join(BASE_DIR, 'models/HQ/best_multimodal_ae_torch_ram.pt'), 
        'tab_scaler': os.path.join(BASE_DIR, 'models/HQ/tab_scaler.joblib'), 
        'npy_scaler': os.path.join(BASE_DIR, 'models/HQ/npy_scaler.joblib') 
    }
}

# 📏 [임계값] (보정된 점수 기준)
THRESHOLDS = {
    # 📌 Tabular AE (Mean: 0.0117 | Std: 0.0049 | Max: 0.1064)
    # loose(2σ): 0.0215, strict(3σ): 0.0266, max: 0.1064
    'tab': {'loose': 0.0215, 'strict': 0.0266, 'max': 0.1064},

    # 📌 RNN AE (Mean: 0.1894 | Std: 0.1748 | Max: 2.6497)
    # loose(2σ): 0.5390, strict(3σ): 0.7137, max: 2.6497
    'rnn': {'loose': 0.5390, 'strict': 0.7137, 'max': 2.6497},

    # 📌 MultiModal AE (Mean: 0.2539 | Std: 0.2012 | Max: 2.0305)
    # loose(2σ): 0.6564, strict(3σ): 0.8600, max: 2.0305
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
# 2. 모델 클래스 정의
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
        laplacian_mean = np.abs(laplacian).mean()
        laplacian_var = laplacian.var()
        light_intensity_mean = region_crop.mean()
        
        light_intensity_change = 0.0
        if prev_region_mean is not None:
            light_intensity_change = light_intensity_mean - prev_region_mean
        
        region_area = (x_max - x_min) * (y_max - y_min)
        
        return {
            'laplacian_mean': laplacian_mean, 'laplacian_var': laplacian_var,
            'light_intensity_mean': light_intensity_mean, 'light_intensity_change': light_intensity_change,
            'region_area': region_area
        }
    except: return None

def detect_speech_segment(video_path, search_start, search_end, progress_callback=None):
    try:
        duration = search_end - search_start
        if duration < VAD_MIN_DURATION: return None, "⚠️ 검색 범위가 너무 짧습니다."

        if progress_callback: progress_callback(0.1)
        y, sr = librosa.load(video_path, sr=VAD_SR, offset=search_start, duration=duration)
        if progress_callback: progress_callback(0.3)
        
        speech_intervals_samples = librosa.effects.split(y, top_db=VAD_TOP_DB, hop_length=512)
        speech_intervals_time = librosa.samples_to_time(speech_intervals_samples, sr=sr)
        
        valid_segments = []
        for start, end in speech_intervals_time:
            if end - start >= VAD_MIN_DURATION:
                valid_segments.append((start + search_start, end + search_start))
        
        if progress_callback: progress_callback(0.5)
        if not valid_segments: return None, "⚠️ 지정된 구간 내에서 뚜렷한 목소리를 찾지 못했습니다."
            
        best_start = valid_segments[0][0]
        best_end = min(valid_segments[0][1], best_start + 5.0) 
        if progress_callback: progress_callback(1.0)
        return (best_start, best_end), f"✅ 분석 구간 확정: {best_start:.2f}초 ~ {best_end:.2f}초"
    except Exception as e: return None, f"❌ 오디오 분석 실패: {str(e)}"

@st.cache_resource
def load_models():
    models = {}
    models['xgb'] = xgb.Booster(); models['xgb'].load_model(MODEL_PATHS['HQ']['xgb'])
    models['tab_ae'] = TabularAE(120, 64).to(device); models['tab_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['tab_ae'], map_location=device)); models['tab_ae'].eval()
    models['rnn_ae'] = RNNAE('GRU', 128, 2, 5).to(device); models['rnn_ae'].load_state_dict(torch.load(MODEL_PATHS['HQ']['rnn_ae'], map_location=device)); models['rnn_ae'].eval()
    models['tab_scaler'] = joblib.load(MODEL_PATHS['HQ']['tab_scaler'])
    models['npy_scaler'] = joblib.load(MODEL_PATHS['HQ']['npy_scaler'])
    return models

def load_multimodal_model():
    cfg = SimpleNamespace(cnn_latent_dim=64, rnn_units=64, bottleneck_dim=64, rnn_model='LSTM')
    model = MultiModalAutoencoder(cfg).to(device)
    model.load_state_dict(torch.load(MODEL_PATHS['HQ']['multi_ae'], map_location=device))
    model.eval()
    return model

# ============================================================
# 4. 핵심 로직 (통계 추출 강화)
# ============================================================

def draw_landmarks_realtime(frame, shape):
    overlay = frame.copy()
    regions = {'eyes': list(range(36, 48)), 'nose': list(range(27, 36)), 'mouth': list(range(48, 68)), 'jaw': list(range(0, 17))}
    colors = {'eyes': (255,255,0), 'nose': (0,255,0), 'mouth': (0,0,255), 'jaw': (255,0,0)}
    for name, indices in regions.items():
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in indices], np.int32)
        cv2.polylines(overlay, [pts], (name != 'jaw'), colors[name], 2)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame

def extract_features(video_path, start_time, end_time, use_audio, progress_callback=None, visualization_callback=None):
    # dlib 경로 확인 및 인코딩 처리
    if not os.path.exists(DLIB_PATH):
        raise FileNotFoundError(f"dlib 파일을 찾을 수 없습니다: {DLIB_PATH}")
    
    # Windows 한글 경로 문제 해결: bytes로 변환
    try:
        dlib_path_bytes = DLIB_PATH.encode('utf-8').decode('utf-8')
    except:
        dlib_path_bytes = DLIB_PATH
    
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(str(DLIB_PATH))
    except RuntimeError as e:
        # 한글 경로 문제일 경우 에러 메시지 개선
        raise RuntimeError(f"dlib 파일 로드 실패. 경로에 한글이 포함되어 문제가 발생했을 수 있습니다.\n경로: {DLIB_PATH}\n원본 에러: {str(e)}")
    
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
    VIZ_INTERVAL = 3 

    while processed_count < total_frames:
        ret, frame = cap.read()
        if not ret: break
        processed_count += 1
        if progress_callback: progress_callback(processed_count / total_frames * 0.7)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_values.append(curr_sharpness)
        
        faces = detector(gray)
        if len(faces) > 0:
            face = faces[0]
            shape = predictor(gray, face)
            if visualization_callback and (processed_count % VIZ_INTERVAL == 0):
                vis_frame = draw_landmarks_realtime(frame.copy(), shape)
                visualization_callback(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            
            frame_feat = {}
            for region_name, indices in FACIAL_LANDMARKS.items():
                prev_mean = prev_regions.get(region_name)
                region_feat = calculate_region_features(gray, shape, region_name, indices, prev_mean)
                if region_feat:
                    frame_feat[region_name] = region_feat
                    prev_regions[region_name] = region_feat['light_intensity_mean']
            if frame_feat: frame_features.append(frame_feat)
    
    cap.release()
    
    # 📊 상세 통계 계산
    input_stats = {
        "mean": np.mean(sharpness_values) if sharpness_values else 0.0,
        "std": np.std(sharpness_values) if sharpness_values else 0.0,
        "min": np.min(sharpness_values) if sharpness_values else 0.0,
        "max": np.max(sharpness_values) if sharpness_values else 0.0
    }
    
    if not frame_features: return None, None, None, input_stats
    
    tab_features = aggregate_tabular_features(frame_features)
    npy_features = create_npy_features(frame_features)
    audio_features = None
    if use_audio:
        if progress_callback: progress_callback(0.8)
        audio_features = extract_audio_features(video_path, start_time, end_time)
        if progress_callback: progress_callback(1.0)
    
    return tab_features, npy_features, audio_features, input_stats

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
        model = whisper.load_model(WHISPER_SIZE)
        model.transcribe(video_path)
        y, sr = librosa.load(video_path, sr=16000, duration=end_time-start_time, offset=start_time)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
        mel = cv2.resize(mel, (128,128))
        return ((mel - mel.min()) / (mel.max() - mel.min())).reshape(1,1,128,128)
    except: return None

def predict(models, features, use_audio):
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
        aud_t = torch.FloatTensor(audio_features).to(device)
        with torch.no_grad():
            img_r, npy_r = multi_model(aud_t, npy_tensor)
            scores['multi'] = (torch.mean((aud_t - img_r)**2) + torch.mean((npy_tensor - npy_r)**2)).item()
    else: scores['multi'] = 0.0
    return scores

def calculate_result(scores, input_stats, sensitivity_k, use_audio):
    """
    [최종 로직] 민감도 상수(K) 기반 보정 (Sensitivity Correction)
    Factor = 입력 선명도 / K (단, 최소 1.0 유지)
    """
    report_lines = []
    th = THRESHOLDS
    
    # 1. 보정 계수 산출 (Safety Clamp: max(1.0, ...))
    sharp_factor = input_stats['mean'] / sensitivity_k
    final_factor = max(1.0, sharp_factor)
    
    # 2. 점수 보정
    adj_scores = {
        'xgb': scores['xgb'],
        'rnn': scores['rnn'] / final_factor,
        'multi': scores['multi'] / final_factor
    }

    def get_status(score, t):
        if score <= t['loose']: return 0.0, "✅ 정상"
        elif score <= t['strict']: return 0.3, "⚠️ 주의"
        elif score <= t['max']: return 0.7, "🔴 의심"
        return 1.0, "🚨 높음"

    report_lines.append(f"**XGBoost**: {adj_scores['xgb']:.4f}")
    p_rnn, s_rnn = get_status(adj_scores['rnn'], th['rnn'])
    report_lines.append(f"**RNN Model**: {adj_scores['rnn']:.2f} (Raw: {scores['rnn']:.0f}) → {s_rnn}")
    
    if use_audio and scores['multi'] > 0:
        p_multi, s_multi = get_status(adj_scores['multi'], th['multi'])
        report_lines.append(f"**Multi Model**: {adj_scores['multi']:.2f} (Raw: {scores['multi']:.0f}) → {s_multi}")
        final = (adj_scores['xgb']*WEIGHTS['xgb']) + (p_rnn*WEIGHTS['rnn']) + (p_multi*WEIGHTS['multi'])
    else:
        report_lines.append("**Multi Model**: OFF")
        final = (adj_scores['xgb']*0.4) + (p_rnn*0.6)

    return {
        'final_prob': final*100, 
        'details': "\n\n".join(report_lines), 
        'scale_factor': final_factor,
        'adj_scores': adj_scores,
        'raw_scores': scores
    }

# ============================================================
# 5. Streamlit UI
# ============================================================
def main():
    st.set_page_config(page_title="Deepfake Detective", page_icon="🕵️", layout="wide")
    
    st.title("🕵️ 딥페이크 탐지 시스템 (HQ 모델)")
    st.markdown("""
    고화질 모델을 사용한 비디오 진위 분석 시스템
    """)
    
    # ============================================================
    # 🔐 로그인 섹션 (Sidebar)
    # ============================================================
    with st.sidebar:
        st.header("🔐 로그인")
        
        if not st.session_state.is_logged_in:
            # 탭: 로그인 / 회원가입
            auth_tab1, auth_tab2 = st.tabs(["로그인", "회원가입"])
            
            with auth_tab1:
                st.subheader("로그인")
                login_email = st.text_input("이메일", key="login_email")
                login_password = st.text_input("비밀번호", type="password", key="login_password")
                
                if st.button("로그인 🔓", key="login_btn"):
                    if login_email and login_password:
                        try:
                            response = requests.post(
                                f"{BACKEND_API_URL}/auth/login",
                                json={"email": login_email, "password": login_password},
                                timeout=10
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.user_id = data.get("id")  # "user_id"가 아니라 "id" 필드 사용
                                st.session_state.email = login_email
                                st.session_state.is_logged_in = True
                                st.success(f"✅ {login_email}님 로그인 성공!")
                                st.rerun()
                            else:
                                st.error(f"❌ 로그인 실패: {response.json().get('detail', '확인되지 않은 사용자')}")
                        except Exception as e:
                            st.error(f"❌ 서버 연결 오류: {str(e)}")
                    else:
                        st.warning("이메일과 비밀번호를 입력해주세요")
            
            with auth_tab2:
                st.subheader("회원가입")
                signup_email = st.text_input("이메일", key="signup_email")
                signup_password = st.text_input("비밀번호", type="password", key="signup_password")
                signup_password_confirm = st.text_input("비밀번호 확인", type="password", key="signup_password_confirm")
                
                if st.button("회원가입 📝", key="signup_btn"):
                    if signup_email and signup_password and signup_password_confirm:
                        if signup_password != signup_password_confirm:
                            st.error("비밀번호가 일치하지 않습니다")
                        else:
                            try:
                                response = requests.post(
                                    f"{BACKEND_API_URL}/auth/signup",
                                    json={"email": signup_email, "password": signup_password},
                                    timeout=10
                                )
                                if response.status_code == 201:
                                    st.success("✅ 회원가입 성공! 로그인 탭에서 로그인해주세요")
                                else:
                                    st.error(f"❌ 회원가입 실패: {response.json().get('detail', '오류 발생')}")
                            except Exception as e:
                                st.error(f"❌ 서버 연결 오류: {str(e)}")
                    else:
                        st.warning("모든 필드를 입력해주세요")
        
        else:
            # 로그인된 상태
            st.success(f"✅ {st.session_state.email}님 로그인됨")
            st.write(f"**User ID**: {st.session_state.user_id}")
            
            if st.button("로그아웃 🔒"):
                st.session_state.user_id = None
                st.session_state.email = None
                st.session_state.is_logged_in = False
                st.info("로그아웃 되었습니다")
                st.rerun()
        
        st.markdown("---")
        st.header("⚙️ 설정")
        # [핵심] 민감도 조절 (기본값 2.0)
        sensitivity_k = st.slider("Sensitivity (K)", 0.1, 10.0, 2.0, 0.1, 
                                help="보정 강도 조절: 낮을수록 점수를 더 많이 깎습니다.")
        
        st.markdown("---")
        if all([os.path.exists(DLIB_PATH), os.path.exists(MODEL_PATHS['HQ']['xgb'])]): st.success("✅ 모델 파일 확인 완료")
        else: st.error("❌ System Offline"); st.stop()
        
        st.markdown("""
        ### 📋 사용법
        1. 비디오 파일 업로드 또는 YouTube 링크 입력
        2. 분석 구간 선택 (파일 업로드 시)
        3. 옵션 설정
        4. 분석 시작 버튼 클릭
        """)

    # ============================================================
    # 📌 입력 모드 선택 (파일 업로드 또는 YouTube)
    # ============================================================
    st.markdown("---")
    input_mode = st.radio(
        "📥 입력 방식 선택",
        ["파일 업로드", "YouTube 링크"],
        horizontal=True
    )

    # 1️⃣ 파일 업로더 (상단에 넓게 배치)
    uploaded_file = None
    youtube_url = None
    
    if input_mode == "파일 업로드":
        uploaded_file = st.file_uploader("📁 비디오 파일 선택", type=['mp4', 'avi', 'mkv', 'mov'], help="Limit 200MB per file")
    else:
        youtube_url = st.text_input("🎥 YouTube URL 입력", placeholder="https://www.youtube.com/watch?v=...")
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # 비디오 메타데이터 추출
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # 화질 등급 판단
        if height >= 1080: quality_label = "🟢 HIGH QUALITY (고화질)"
        elif height >= 720: quality_label = "🟡 MEDIUM QUALITY (중화질)"
        else: quality_label = "🔴 LOW QUALITY (저화질)"

        # 2️⃣ 미리보기 및 정보 레이아웃 (좌: 영상 / 우: 정보)
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("📹 비디오 미리보기")
            st.video(uploaded_file)
            
        with col2:
            st.subheader("📊 영상 정보")
            
            # 정보 테이블 데이터 생성
            video_info_data = {
                "항목": ["해상도", "FPS", "길이", "프레임 수", "화질"],
                "값": [f"{width} x {height}", f"{fps:.1f}", f"{duration:.2f}초", f"{frame_count}", quality_label]
            }
            df_info = pd.DataFrame(video_info_data)
            
            # 테이블 표시
            st.table(df_info.set_index("항목"))
            
            # 분석 제어 위젯을 정보 하단에 배치
            st.markdown("---")
            search_range = st.slider("탐색 범위 (초)", 0.0, duration, (0.0, min(duration, 15.0)))
            start_btn = st.button("🚀 분석 시작", type="primary", use_container_width=True)

        if start_btn:
            s_start, s_end = search_range
            if s_end - s_start < 3.0: st.toast("⚠️ 3초 이상 선택하세요", icon="⚠️"); st.stop()
            
            st.markdown("---")
            mon_col1, mon_col2 = st.columns([1, 1])
            with mon_col1: prog = st.progress(0); stat = st.empty()
            with mon_col2: img_ph = st.empty()

            try:
                stat.text("🔉 VAD 탐색 중...")
                vad, msg = detect_speech_segment(video_path, s_start, s_end, lambda x: prog.progress(x*0.1))
                if not vad: st.error(msg); st.stop()
                f_start, f_end = vad
                
                stat.text("📦 모델 로드 중..."); models = load_models(); prog.progress(0.15)
                
                stat.text("🎬 특징 추출 중...")
                feat = extract_features(video_path, f_start, f_end, True, lambda x: prog.progress(0.15+(x*0.65)), lambda f: img_ph.image(f, use_container_width=True))
                tab, npy, aud, input_stats = feat
                
                if tab is None: st.error("❌ 얼굴을 찾을 수 없습니다."); st.stop()

                stat.text("🧠 패턴 분석 및 보정 중...")
                raw = predict(models, (tab, npy, aud), True)
                
                # [최종 결과 계산]
                res = calculate_result(raw, input_stats, sensitivity_k, True)
                
                prog.progress(1.0); stat.text("✅ 분석 완료")
                
                # 📊 [Dashboard]
                st.markdown("### 📊 Analysis Dashboard")
                d1, d2, d3 = st.columns(3)
                d1.metric("Input Sharpness", f"{input_stats['mean']:.1f}", help="입력 영상의 평균 선명도")
                d2.metric("Sensitivity (K)", f"{sensitivity_k}", help="사용자가 설정한 민감도 상수")
                d3.metric("Correction Factor", f"x{res['scale_factor']:.2f}", help="최종적으로 점수를 나눈 값")
                
                st.divider()
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    if res['final_prob'] >= 50: st.error(f"### 🚨 FAKE\n**{res['final_prob']:.2f}%**")
                    else: st.success(f"### ✅ REAL\n**{res['final_prob']:.2f}%**")
                with c2:
                    st.markdown("#### 📝 판정 요약")
                    st.markdown(res['details'])
                
                with st.expander("🔍 Raw Data (Debug)"):
                    st.json(res['raw_scores'])
                    st.json(input_stats)
                    
            except Exception as e: st.error(f"Error: {e}")
            finally: 
                try: os.unlink(video_path)
                except: pass
    
    # ============================================================
    # 🎥 YouTube 링크 처리
    # ============================================================
    elif youtube_url:
        if not st.session_state.is_logged_in:
            st.warning("⚠️ YouTube 분석을 위해 먼저 로그인해주세요")
        else:
            if st.button("🚀 YouTube 비디오 분석 시작", type="primary", use_container_width=True):
                try:
                    st.info("⏳ YouTube 비디오를 다운로드 중입니다...")
                    
                    # 백엔드 API 호출
                    response = requests.post(
                        f"{BACKEND_API_URL}/detect/youtube",
                        json={
                            "url": youtube_url,
                            "user_id": st.session_state.user_id,
                            "sensitivity_k": sensitivity_k
                        },
                        timeout=300  # 5분 타임아웃
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ 분석 완료!")
                        
                        # 📊 Dashboard 표시
                        st.markdown("### 📊 Analysis Dashboard")
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Fake Probability", f"{result.get('fake_probability', 0):.2f}%")
                        d2.metric("Is Fake", "🚨 FAKE" if result.get('is_fake') else "✅ REAL")
                        d3.metric("Sensitivity (K)", f"{sensitivity_k}")
                        
                        st.divider()
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            if result.get('is_fake'):
                                st.error(f"### 🚨 FAKE\n**{result.get('fake_probability', 0):.2f}%**")
                            else:
                                st.success(f"### ✅ REAL\n**{result.get('fake_probability', 0):.2f}%**")
                        
                        with c2:
                            st.markdown("#### 📝 분석 결과")
                            st.markdown(f"""
                            - **Fake Probability**: {result.get('fake_probability', 0):.2f}%
                            - **Input Sharpness**: {result.get('input_sharpness', 'N/A')}
                            - **Video ID**: {result.get('video_id', 'N/A')}
                            """)
                        
                        with st.expander("🔍 상세 점수"):
                            scores = result.get('scores', {})
                            st.json(scores)
                    
                    elif response.status_code == 400:
                        error = response.json()
                        st.error(f"❌ 분석 오류: {error.get('detail', '알 수 없는 오류')}")
                    else:
                        st.error(f"❌ 서버 오류: {response.status_code}")
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ 요청 시간이 초과되었습니다. 나중에 다시 시도해주세요")
                except requests.exceptions.ConnectionError:
                    st.error("❌ 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해주세요")
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()