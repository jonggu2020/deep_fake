import pandas as pd
import numpy as np
import cv2  # (pip install opencv-python)
import os
import glob
import time

# --- 1. 사용자 설정 ---

# ⚠️ [조정 필요] 표준 편차(Std) 임계값
# 이 값보다 Std가 낮으면 '의심' 파일로 분류됩니다.ㄴ
# (이전의 '침묵' 이미지의 Std 값을 참고하여 40.0 ~ 60.0 사이로 설정하세요)
SUSPICIOUS_THRESHOLD = 45.0 

# ⚠️ [수정필요 1]
# 현재 정제된 CSV 파일 (v9 스크립트로 생성한 최신 CSV)
# 예: "./master_summary_v11_cleaned_final.csv"
CLEANED_CSV_FILE = "./master_summary_v11_cleaned_final.csv" 

# ⚠️ [수정필요 2]
# PNG 파일이 있는 폴더 경로
PNG_DIR = "./3_audio_spectrograms"

# ⚠️ [출력]
# '의심' 파일의 ID 목록이 저장될 텍스트 파일
SUSPECT_LIST_FILE = "./suspect_list.txt"
# ---

def run_phase1_filter(threshold, csv_path, png_dir, output_txt_path):
    """
    (PART 1) 
    CSV에 존재하는 ID를 기준으로 PNG를 분석하여 '의심' 목록을 생성합니다.
    """
    print("="*70)
    print(f"PART 1: '의심' 오디오(PNG) 필터링 시작")
    print(f"         (기준 임계값: Std < {threshold})")
    print("="*70)

    # 1. 유효한 ID 목록을 CSV에서 로드
    try:
        df = pd.read_csv(csv_path)
        if 'video_id' not in df.columns:
            print(f"❌ 오류: CSV에 'video_id' 컬럼이 없습니다.")
            return
        # Set으로 만들어 빠른 조회
        valid_ids = set(df['video_id'])
        print(f"✓ '{csv_path}'에서 {len(valid_ids)}개의 유효 ID 로드 완료.")
    except Exception as e:
        print(f"❌ 오류: CSV 파일 로드 실패: {e}")
        return

    # 2. PNG 폴더 검사
    if not os.path.isdir(png_dir):
        print(f"❌ 오류: PNG 폴더를 찾을 수 없습니다: {png_dir}")
        return

    suspect_ids = []
    
    print(f"\n... {len(valid_ids)}개의 유효 ID에 대해 PNG 파일 분석 중 ...")
    
    # CSV에 있는 ID 목록을 기준으로 순회
    analysis_count = 0
    for video_id in valid_ids:
        png_path = os.path.join(png_dir, f"{video_id}.png")
        
        if not os.path.exists(png_path):
            # CSV에는 있으나 PNG가 없는 경우 (이전 작업 오류)
            # print(f"  [경고] PNG 파일 없음: {video_id}.png (건너뜀)")
            continue

        try:
            # OpenCV로 이미지 로드 (흑백)
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [경고] {video_id}.png 이미지 로드 실패 (손상?)")
                continue
            
            # 픽셀 값의 표준 편차 계산
            file_std = np.std(img)
            
            # 임계값과 비교하여 '의심' 목록에 추가
            if file_std < threshold:
                suspect_ids.append(video_id)
            
            analysis_count += 1
            if analysis_count % 5000 == 0:
                 print(f"  ... {analysis_count} / {len(valid_ids)}개 분석 완료 ...")

        except Exception as e:
            print(f"  ❌ 오류: '{png_path}' 파일 분석 중 오류: {e}")

    # 3. 결과 저장
    print("\n... 분석 완료 ...")
    
    try:
        with open(output_txt_path, 'w') as f:
            for video_id in suspect_ids:
                f.write(f"{video_id}\n")
        
        print(f"✓ '의심' ID 목록을 '{output_txt_path}' 파일에 저장했습니다.")
        
    except Exception as e:
        print(f"❌ 오류: '의심' 목록 파일 저장 실패: {e}")

    print("\n" + "="*70)
    print("🎉 1차 필터링(의심) 완료!")
    print(f"  - 총 분석 대상 (CSV 기준): {len(valid_ids)} 개")
    print(f"  - 실제 분석된 PNG: {analysis_count} 개")
    print(f"\n  >>> '의심'으로 분류된 PNG 파일: {len(suspect_ids)} 개 <<<\n")
    print("="*70)
    print(f"다음 단계: '{SUSPECT_LIST_FILE}' 파일을 기반으로 2차 GUI 검토 스크립트를 실행하세요.")
    
if __name__ == "__main__":
    start_time = time.time()
    run_phase1_filter(
        threshold=SUSPICIOUS_THRESHOLD,
        csv_path=CLEANED_CSV_FILE,
        png_dir=PNG_DIR,
        output_txt_path=SUSPECT_LIST_FILE
    )
    end_time = time.time()

    print(f"총 소요 시간: {end_time - start_time:.2f}초")
