import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from landmark_smoother import LandmarkSmoother  # 방금 만든 스무더 불러오기

# ---------------------------------------------------------
# 1. 모델 정의 (사용자님 모델 구조에 맞게 유지)
# ---------------------------------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

# ---------------------------------------------------------
# 2. [핵심] 스무딩이 포함된 통합 판독 함수
# ---------------------------------------------------------
def predict_video_is_deepfake(csv_path, model, threshold=0.4):
    """
    영상의 랜드마크 CSV를 읽어서 딥페이크 여부를 판독합니다.
    (자동으로 노이즈 제거 스무딩을 적용합니다)
    """
    print(f"[판독 시작] 파일: {csv_path}")

    # A. 데이터 로드
    try:
        df = pd.read_csv(csv_path, header=None, low_memory=False)
        # 숫자 변환 가능한 행만 필터링
        df = df[pd.to_numeric(df[2], errors='coerce').notnull()]
        # 좌표값만 추출 (2번 컬럼부터)
        raw_data = df.iloc[:, 2:].astype(float).values
    except Exception as e:
        print(f"[에러] 데이터 로드 실패: {e}")
        return "Error"

    # ----------------------------------------------------------
    # B. [여기가 핵심!] 노이즈 제거 (Smoothing)
    # ----------------------------------------------------------
    # 그래프에서 확인한 window_size=7 적용
    smoother = LandmarkSmoother(window_size=7)
    
    # 빨간색 데이터를 -> 파란색 데이터로 변환
    clean_data = smoother.apply(raw_data) 
    
    print(f"- 전처리 완료: {len(raw_data)} 프레임에 스무딩 적용됨")
    # ----------------------------------------------------------

    # C. 모델 추론
    device = next(model.parameters()).device
    input_tensor = torch.FloatTensor(clean_data).unsqueeze(0).to(device) # (1, Seq, Dim)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        
        # Loss 계산 (MAE)
        loss = torch.mean(torch.abs(output - input_tensor)).item()

    # D. 결과 판정
    is_deepfake = loss > threshold
    result_str = "딥페이크(Deepfake)" if is_deepfake else "정상(Real)"
    
    print(f"\n[판독 결과]")
    print(f"---------------------------------------")
    print(f"▶ 모델 에러값(Loss): {loss:.4f}")
    print(f"▶ 기준 임계값(Threshold): {threshold:.4f}")
    print(f"---------------------------------------")
    print(f"▶ 최종 판정: {result_str}")
    
    return loss, is_deepfake

# ---------------------------------------------------------
# 사용 예시
# ---------------------------------------------------------
if __name__ == "__main__":
    # 설정
    MODEL_PATH = "rnn_model.pth"
    TEST_CSV = "LQ_CNN.csv" # 테스트할 영상(정상인데 비정상으로 떴던 그 파일)
    INPUT_DIM = 136 # 68 * 2
    HIDDEN_DIM = 128
    LAYERS = 2
    
    # 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoEncoder(INPUT_DIM, HIDDEN_DIM, LAYERS).to(device)
    
    if pd.io.common.file_exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        
        # 판독 실행
        # threshold는 아까 'recalc_threshold.py'로 구한 값을 넣으시면 완벽합니다.
        # (일단 기존 0.4로 테스트해봐도 Loss가 확 떨어지는게 보일 겁니다)
        predict_video_is_deepfake(TEST_CSV, model, threshold=0.4)
    else:
        print("모델 파일이 없습니다. 경로를 확인하세요.")