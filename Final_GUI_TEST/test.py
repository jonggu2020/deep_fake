import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import dlib
import librosa
import os
import tempfile
import joblib
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ==========================================
# 1. 모델 설정 및 클래스 정의 (학습 코드와 100% 동일하게 수정)
# ==========================================
# train_best_model.py의 설정값을 그대로 가져왔습니다.
config = SimpleNamespace(
    batch_size = 32,
    bottleneck_dim = 64,
    cnn_latent_dim = 128, # 학습 코드에 맞춰 128로 설정
    cnn_model = "LeNet",
    rnn_model = "GRU",
    rnn_units = 64
)

IMG_HEIGHT, IMG_WIDTH = 128, 128
NPY_SEQ_LENGTH, NPY_FEATURES = 90, 5

class MultiModalAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(MultiModalAutoencoder, self).__init__()
        
        # --- [수정] 학습 코드의 LeNet 구조 적용 ---
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), # 64x64
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), # 32x32
            nn.Flatten(), 
            nn.Linear(32 * 32 * 32, cfg.cnn_latent_dim), nn.ReLU()
        )

        # --- [수정] 학습 코드의 GRU 구조 적용 ---
        self.rnn_encoder = nn.GRU(input_size=NPY_FEATURES, hidden_size=cfg.rnn_units, batch_first=True)
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.cnn_latent_dim + cfg.rnn_units, cfg.bottleneck_dim), 
            nn.ReLU()
        )
        
        # RNN Decoder (GRU)
        self.rnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, cfg.rnn_units)
        self.rnn_decoder = nn.GRU(input_size=cfg.rnn_units, hidden_size=cfg.rnn_units, batch_first=True)
        self.rnn_output_layer = nn.Linear(cfg.rnn_units, NPY_FEATURES)
        
        # CNN Decoder
        self.cnn_decoder_fc = nn.Linear(cfg.bottleneck_dim, 64 * 16 * 16)
        self.cnn_decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        
    def forward(self, img, npy):
        # Encoding
        cnn_feat = self.cnn_encoder(img)
        _, h_n = self.rnn_encoder(npy) # GRU는 h_n만 반환
        
        # Fusion (h_n[-1] 사용)
        z = self.bottleneck(torch.cat((cnn_feat, h_n[-1]), dim=1))
        
        # Decoding
        rnn_in = self.rnn_decoder_fc(z).unsqueeze(1).repeat(1, NPY_SEQ_LENGTH, 1)
        rnn_out, _ = self.rnn_decoder(rnn_in)
        
        cnn_out = self.cnn_decoder(self.cnn_decoder_fc(z))
        
        return cnn_out, self.rnn_output_layer(rnn_out)

# ==========================================
# 2. 특징 추출 함수 (Matplotlib 버그 수정 포함)
# ==========================================
DLIB_PATH = "shape_predictor_68_face_landmarks.dat"
FACIAL_LANDMARKS = {"mouth": list(range(48, 68))}

def get_region_bounding_box(shape, landmark_indices):
    points = [(shape.part(i).x, shape.part(i).y) for i in landmark_indices]
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))

def extract_features(video_path, start_sec, end_sec, predictor_path):
    if not os.path.exists(predictor_path):
        return None, None, "Dlib 모델 파일을 찾을 수 없습니다."
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    mouth_features = []
    prev_light_mean = None
    frames_to_read = end_frame - start_frame
    
    # 오디오 처리
    try:
        y, sr = librosa.load(video_path, sr=44100, offset=start_sec, duration=(end_sec - start_sec))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        fig.canvas.draw()
        
        # [수정] 최신 matplotlib 대응
        img_np = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
        img_normalized = img_resized / 255.0
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
    except Exception as e:
        return None, None, f"오디오 처리 오류: {str(e)}"

    # 비디오 프레임 처리
    count = 0
    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0: continue
        
        shape = predictor(gray, faces[0])
        indices = FACIAL_LANDMARKS['mouth']
        x1, y1, x2, y2 = get_region_bounding_box(shape, indices)
        
        h, w = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        region = gray[y1:y2, x1:x2]
        if region.size == 0: continue
        
        laplacian = cv2.Laplacian(region, cv2.CV_64F)
        l_mean = np.abs(laplacian).mean()
        l_var = laplacian.var()
        light_mean = region.mean()
        light_change = (light_mean - prev_light_mean) if prev_light_mean is not None else 0.0
        prev_light_mean = light_mean
        area = (x2 - x1) * (y2 - y1)
        
        mouth_features.append([l_mean, l_var, light_mean, light_change, area])
        count += 1

    cap.release()
    if count < 10:
        return None, None, "얼굴이 감지된 프레임이 너무 적습니다."

    return img_tensor, np.array(mouth_features), None

# ==========================================
# 3. Streamlit UI (캘리브레이션 모드)
# ==========================================
st.set_page_config(page_title="딥페이크 탐지기", layout="wide")
st.title("🕵️‍♂️ 딥페이크 탐지기 (LeNet+GRU Model)")
st.markdown("학습된 **LeNet + GRU** 모델을 사용하여 오디오/비디오 부조화를 분석합니다.")

# 사이드바
with st.sidebar:
    st.header("설정")
    # 사용자가 직접 값을 보고 조절할 수 있도록 입력창 제공
    threshold = st.number_input("의심 기준값 (Threshold)", value=0.0050, format="%.6f", step=0.0001)
    st.info(f"Loss 값이 **{threshold:.6f}** 보다 크면 딥페이크로 판단합니다.")

uploaded_file = st.file_uploader("비디오 업로드", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 원본 영상")
        st.video(uploaded_file)
    
    with col2:
        st.subheader("2. 분석 구간")
        range_val = st.slider("구간 선택 (초)", 0.0, duration, (0.0, min(duration, 5.0)))
        start_sec, end_sec = range_val
        
        if st.button("🚀 분석 시작"):
            if end_sec - start_sec < 1.0:
                st.error("최소 1초 이상 선택해주세요.")
            else:
                with st.spinner('분석 중... (모델 로딩 및 특징 추출)'):
                    try:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        
                        # 모델 로드
                        model = MultiModalAutoencoder(config).to(device)
                        # [주의] 파일명이 train_best_model.py에서 저장한 이름과 같은지 확인하세요
                        model.load_state_dict(torch.load("best_deepfake_model.pt", map_location=device))
                        model.eval()
                        
                        # Scaler 로드 (파일명 확인 필요: npy_scaler_final.joblib 또는 npy_scaler.joblib)
                        scaler_path = "npy_scaler_final.joblib"
                        if not os.path.exists(scaler_path):
                             scaler_path = "npy_scaler.joblib" # 이름이 다를 경우 대비
                        scaler = joblib.load(scaler_path)
                        
                        # 특징 추출
                        img_tensor, npy_raw, err = extract_features(video_path, start_sec, end_sec, DLIB_PATH)
                        
                        if err:
                            st.error(err)
                        else:
                            # 전처리
                            npy_scaled = scaler.transform(npy_raw)
                            curr = npy_scaled.shape[0]
                            padded = np.zeros((NPY_SEQ_LENGTH, NPY_FEATURES))
                            if curr > NPY_SEQ_LENGTH: padded = npy_scaled[:NPY_SEQ_LENGTH, :]
                            else: padded[:curr, :] = npy_scaled
                            
                            npy_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                            img_tensor = img_tensor.to(device)
                            
                            # 추론
                            with torch.no_grad():
                                r_img, r_npy = model(img_tensor, npy_tensor)
                                l_img = nn.MSELoss()(r_img, img_tensor).item()
                                l_npy = nn.MSELoss()(r_npy, npy_tensor).item()
                                total_loss = l_img + l_npy
                            
                            # 결과 표시
                            st.divider()
                            st.metric(label="총 복원 오차 (Loss)", value=f"{total_loss:.6f}")
                            
                            c1, c2 = st.columns(2)
                            c1.info(f"🖼️ 이미지 오차: {l_img:.6f}")
                            c2.info(f"📈 시계열 오차: {l_npy:.6f}")
                            
                            # 시각적 판단
                            diff = total_loss - threshold
                            if total_loss > threshold:
                                st.error(f"🚨 **Deepfake 의심** (기준보다 +{diff:.6f} 높음)")
                            else:
                                st.success(f"✅ **정상(Real) 추정** (기준보다 {diff:.6f} 낮음)")

                    except Exception as e:
                        st.error(f"오류 발생: {e}")
                        st.warning("팁: 모델 파일명(best_deepfake_model.pt)이나 Scaler 파일명이 폴더에 있는지 확인하세요.")