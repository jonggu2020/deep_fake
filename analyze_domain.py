import cv2
import numpy as np
import os
import json
from tqdm import tqdm

# ==========================================
# ⚙️ 설정: 학습 데이터(원본) 폴더 경로
# ==========================================
TRAIN_DATA_DIR = "./data"  # 학습 데이터 폴더
OUTPUT_CONFIG_FILE = "domain_config.json"
SAMPLE_COUNT = 500  # 100개 정도만 샘플링하면 충분함

def analyze_domain(video_folder, max_samples=None):
    print(f"🔍 학습 데이터 도메인 분석 시작: {video_folder}")
    
    video_files = []
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print("❌ 비디오 파일이 없습니다.")
        return None

    # 랜덤 샘플링
    if max_samples and len(video_files) > max_samples:
        import random
        video_files = random.sample(video_files, max_samples)

    widths = []
    heights = []
    laplacian_vars = []

    for video_path in tqdm(video_files):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        # 영상 중간 프레임 추출
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        
        if ret:
            h, w = frame.shape[:2]
            widths.append(w)
            heights.append(h)
            
            # [중요] 선명도(Laplacian Variance) 측정
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacian_vars.append(lap_var)
            
        cap.release()

    # 통계 요약
    stats = {
        "target_width": int(np.mean(widths)),           # 평균 너비 (아마 426)
        "target_height": int(np.mean(heights)),         # 평균 높이 (아마 240)
        "avg_laplacian": float(np.mean(laplacian_vars)), # ★ 타겟 선명도 (이게 기준값!)
        "std_laplacian": float(np.std(laplacian_vars))
    }
    
    return stats

if __name__ == "__main__":
    stats = analyze_domain(TRAIN_DATA_DIR, SAMPLE_COUNT)
    
    if stats:
        print("\n📊 [분석 결과 - 이 값이 기준이 됩니다]")
        print(f"  - 타겟 해상도: {stats['target_width']} x {stats['target_height']}")
        print(f"  - 타겟 선명도: {stats['avg_laplacian']:.2f}")
        
        with open(OUTPUT_CONFIG_FILE, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"✅ 설정 파일 저장 완료: {OUTPUT_CONFIG_FILE}")