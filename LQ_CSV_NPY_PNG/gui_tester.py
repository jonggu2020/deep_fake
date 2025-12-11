import os
# ============================================================
# ⚠️ [수정됨] OpenMP 충돌 해결 (반드시 최상단 위치)
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import librosa
import librosa.display
from torchvision import transforms, models

# ============================================================
# 1. 모델 아키텍처 정의 (학습 코드와 동일해야 함)
# ============================================================
class EfficientNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # --- Encoder (Pre-trained EfficientNet-B0) ---
        # weights=None으로 설정하여 경고 메시지 제거 (어차피 로드하므로)
        efficientnet = models.efficientnet_b0(weights=None) 
        self.encoder_features = efficientnet.features
        
        self.encoder_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, latent_dim),
            nn.ReLU()
        )
        
        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, 1280 * 4 * 4)
        
        self.decoder_layers = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (1280, 4, 4)),
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder_features(x)
        latent = self.encoder_head(x)
        x = self.decoder_input(latent)
        reconstructed = self.decoder_layers(x)
        return reconstructed

# ============================================================
# 2. GUI 애플리케이션 클래스
# ============================================================
class AudioDeepfakeTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Audio Anomaly Detection Tester")
        self.root.geometry("900x750")
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Device: {self.device}")

        # 모델 로드
        self.model = self.load_model("best_model_final.pth")
        
        # 변수 초기화
        self.video_path = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)), # 학습 설정과 동일
            transforms.ToTensor(),
        ])

        # UI 구성
        self.create_widgets()

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"모델 파일을 찾을 수 없습니다:\n{model_path}")
            return None
        
        try:
            model = EfficientNetAutoencoder(latent_dim=256).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            print("✅ 모델 로드 완료")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"모델 로드 중 오류 발생:\n{e}")
            return None

    def create_widgets(self):
        # 1. 파일 선택 영역
        frame_top = tk.Frame(self.root, pady=10)
        frame_top.pack(fill="x", padx=20)
        
        self.btn_load = tk.Button(frame_top, text="📂 비디오 파일 열기", command=self.select_video, bg="#e1e1e1", font=("Arial", 12))
        self.btn_load.pack(side="left")
        
        self.lbl_filepath = tk.Label(frame_top, text="선택된 파일 없음", fg="gray", font=("Arial", 10))
        self.lbl_filepath.pack(side="left", padx=10)

        # 2. 설정 영역 (시작/종료 시간)
        frame_controls = tk.Frame(self.root, pady=10)
        frame_controls.pack(fill="x", padx=20)
        
        tk.Label(frame_controls, text="시작 시간(초):").pack(side="left")
        self.entry_start = tk.Entry(frame_controls, width=8)
        self.entry_start.insert(0, "0.0")
        self.entry_start.pack(side="left", padx=5)
        
        tk.Label(frame_controls, text="종료 시간(초):").pack(side="left", padx=(10, 0))
        self.entry_end = tk.Entry(frame_controls, width=8)
        self.entry_end.insert(0, "3.0")
        self.entry_end.pack(side="left", padx=5)

        self.btn_run = tk.Button(frame_controls, text="🚀 분석 시작", command=self.run_inference, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.btn_run.pack(side="left", padx=20)

        # 3. 결과 텍스트 영역
        frame_result = tk.Frame(self.root, pady=10, bg="#f0f0f0")
        frame_result.pack(fill="x", padx=20)
        
        self.lbl_loss = tk.Label(frame_result, text="Reconstruction Loss: -", font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.lbl_loss.pack(pady=5)
        
        self.lbl_desc = tk.Label(frame_result, text="(Loss가 낮을수록 학습된 데이터(Real)와 유사함)", font=("Arial", 10), bg="#f0f0f0", fg="gray")
        self.lbl_desc.pack(pady=2)

        # 4. 이미지 시각화 영역 (Canvas)
        frame_images = tk.Frame(self.root)
        frame_images.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 원본 이미지
        self.panel_orig = tk.Label(frame_images, text="Original Spectrogram")
        self.panel_orig.pack(side="left", expand=True, fill="both")
        
        # 복원 이미지
        self.panel_recon = tk.Label(frame_images, text="Reconstructed Spectrogram")
        self.panel_recon.pack(side="right", expand=True, fill="both")

    def select_video(self):
        path = filedialog.askopenfilename(
            title="비디오 파일 선택",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        if path:
            self.video_path = path
            self.lbl_filepath.config(text=os.path.basename(path), fg="black")

    def get_spectrogram_image(self, audio_path, start_sec, end_sec):
        """
        학습 데이터 생성 로직과 100% 동일하게 메모리 상에서 이미지를 생성합니다.
        """
        try:
            # 1. 오디오 로드 (librosa)
            duration = end_sec - start_sec
            # warnings를 일시적으로 무시하거나 soundfile 사용 유도
            y, sr = librosa.load(audio_path, sr=44100, offset=start_sec, duration=duration)
            
            if len(y) == 0:
                return None, "오디오 데이터가 비어있습니다."

            # 2. 스펙트로그램 변환
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # 3. Matplotlib로 이미지 그리기 (메모리 버퍼 사용)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            
            # 4. 버퍼를 OpenCV 포맷으로 변환
            file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB (학습 코드와 일치)
            
            return img, None
            
        except Exception as e:
            return None, str(e)

    def run_inference(self):
        if not self.model:
            messagebox.showerror("Error", "모델이 로드되지 않았습니다.")
            return
        if not self.video_path:
            messagebox.showwarning("Warning", "비디오 파일을 선택해주세요.")
            return

        try:
            start = float(self.entry_start.get())
            end = float(self.entry_end.get())
            if start >= end:
                messagebox.showwarning("Warning", "시작 시간은 종료 시간보다 작아야 합니다.")
                return
        except ValueError:
            messagebox.showwarning("Warning", "시간은 숫자여야 합니다.")
            return

        # 1. 전처리 (이미지 생성)
        self.btn_run.config(state="disabled", text="분석 중...")
        self.root.update()

        img_rgb, error = self.get_spectrogram_image(self.video_path, start, end)
        
        if img_rgb is None:
            messagebox.showerror("Error", f"전처리 실패: {error}")
            self.btn_run.config(state="normal", text="🚀 분석 시작")
            return

        # 2. 텐서 변환
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device) # (1, 3, 128, 128)

        # 3. 모델 추론
        with torch.no_grad():
            reconstructed = self.model(input_tensor)
            loss = nn.MSELoss()(reconstructed, input_tensor).item()

        # 4. 결과 시각화
        self.display_results(input_tensor, reconstructed, loss)
        self.btn_run.config(state="normal", text="🚀 분석 시작")

    def display_results(self, original_tensor, recon_tensor, loss):
        # Loss 표시
        loss_str = f"{loss:.6f}"
        self.lbl_loss.config(text=f"Reconstruction Loss: {loss_str}")
        
        # Loss 색상 코딩 (단순 예시 기준, 실제 데이터에 따라 조정 필요)
        # Autoencoder에서 학습하지 않은 데이터(Fake)는 Loss가 높음
        if loss > 0.01: # 임의의 임계값 (사용자가 테스트하며 감 잡아야 함)
            self.lbl_loss.config(fg="red")
            self.lbl_desc.config(text="높은 오차: 학습 데이터와 다름 (잠재적 Fake/Anomaly)")
        else:
            self.lbl_loss.config(fg="green")
            self.lbl_desc.config(text="낮은 오차: 학습 데이터와 유사함 (Real)")

        # Tensor -> PIL Image 변환
        to_pil = transforms.ToPILImage()
        
        orig_img = to_pil(original_tensor.squeeze().cpu())
        recon_img = to_pil(recon_tensor.squeeze().cpu())

        # 이미지 크기 조정 (화면에 맞게)
        disp_size = (400, 400)
        orig_img = orig_img.resize(disp_size)
        recon_img = recon_img.resize(disp_size)

        # Tkinter 이미지 객체 생성
        self.tk_orig = ImageTk.PhotoImage(orig_img)
        self.tk_recon = ImageTk.PhotoImage(recon_img)

        # 라벨에 이미지 및 텍스트 업데이트
        self.panel_orig.config(image=self.tk_orig, text="[입력] Original Spectrogram", compound="top", font=("Arial", 12, "bold"))
        self.panel_orig.image = self.tk_orig # 참조 유지
        
        self.panel_recon.config(image=self.tk_recon, text="[복원] Autoencoder Reconstruction", compound="top", font=("Arial", 12, "bold"))
        self.panel_recon.image = self.tk_recon # 참조 유지


# ============================================================
# 3. 메인 실행
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioDeepfakeTesterApp(root)
    root.mainloop()