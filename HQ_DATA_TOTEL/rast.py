import pandas as pd
import os
import glob
import shutil
import time
from tqdm import tqdm # (pip install tqdm)

# --- 1. 사용자 설정 ---

# ⚠️ [수정필요 1]
# 수동 검수가 완료된 "기준" PNG 폴더 (28,828개 파일)
PNG_SOURCE_DIR = "./3_audio_spectrograms"

# ⚠️ [수정필요 2]
# 동기화할 원본 NPY 폴더 (모든 NPY 파일이 있는 곳)
ORIGINAL_NPY_DIR = "./2_npy_timeseries"

# ⚠️ [수정필요 3]
# 동기화할 원본 CSV 파일 (모든 행이 있는 곳)
# 예: "./master_summary_v11_cleaned_final.csv"
ORIGINAL_CSV_FILE = "./master_summary_v12_audio_cleaned.csv"

# --- 2. 출력 경로 설정 ---

# ⚠️ [출력 1]
# 최종 28,828개의 NPY 파일만 복사될 "새 폴더"
FINAL_NPY_DIR = "./FINAL_NPY_28828"

# ⚠️ [출력 2]
# 최종 28,828개의 행만 필터링된 "새 CSV 파일"
FINAL_CSV_FILE = "./FINAL_master_summary_28828.csv"

# ---

def get_master_id_list(png_dir):
    """
    기준 폴더(PNG)에서 28,828개의 파일 ID 목록(Set)을 생성합니다.
    """
    print("="*70)
    print(f"PART 1: 기준 ID 목록 생성 시작")
    print(f"         (소스: '{png_dir}')")
    print("="*70)
    
    if not os.path.isdir(png_dir):
        print(f"❌ 오류: 기준 PNG 폴더를 찾을 수 없습니다: {png_dir}")
        return None
        
    png_files = glob.glob(os.path.join(png_dir, "*.png"))
    
    if not png_files:
        print(f"❌ 오류: '{png_dir}'에 PNG 파일이 없습니다.")
        return None
        
    # 파일명에서 ID(확장자 제외)만 추출하여 Set으로 만듦
    master_id_set = {os.path.splitext(os.path.basename(f))[0] for f in png_files}
    
    print(f"✓ 총 {len(master_id_set)}개의 고유한 ID(기준)를 로드했습니다.")
    
    # 28,828개가 맞는지 확인
    if len(master_id_set) != 28828:
        print(f"  ⚠️ 경고: PNG 파일 개수가 28,828개가 아닌 {len(master_id_set)}개입니다.")
        print("         일단 이 개수를 기준으로 동기화를 진행합니다.")
        
    return master_id_set

def sync_npy_files(master_ids, original_dir, final_dir):
    """
    (PART 2)
    원본 NPY 폴더에서 master_ids에 해당하는 파일만
    FINAL_NPY_DIR로 '복사'합니다.
    """
    print("\n" + "="*70)
    print(f"PART 2: NPY 파일 동기화 (복사) 시작")
    print(f"         (소스: '{original_dir}')")
    print(f"         (대상: '{final_dir}')")
    print("="*70)

    if not os.path.isdir(original_dir):
        print(f"❌ 오류: 원본 NPY 폴더를 찾을 수 없습니다: {original_dir}")
        return False
        
    os.makedirs(final_dir, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    print(f"총 {len(master_ids)}개의 NPY 파일을 복사합니다...")
    
    # tqdm을 사용하여 진행률 표시
    for video_id in tqdm(master_ids, desc="NPY 파일 복사 중"):
        source_path = os.path.join(original_dir, f"{video_id}.npy")
        dest_path = os.path.join(final_dir, f"{video_id}.npy")
        
        if os.path.exists(source_path):
            try:
                # shutil.copy2는 메타데이터까지 복사합니다.
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"  - 복사 실패: {video_id} ({e})")
        else:
            print(f"  - 원본 없음 (Missing): {video_id}.npy")
            missing_count += 1

    print("\n✓ NPY 파일 동기화 완료.")
    print(f"  - 성공: {copied_count} 개")
    print(f"  - 원본 경로에 파일이 없어 실패: {missing_count} 개")
    print(f"  - 최종 폴더: '{final_dir}'")
    return True

def sync_csv_file(master_ids, original_csv, final_csv):
    """
    (PART 3)
    원본 CSV 파일에서 master_ids에 해당하는 행만
    필터링하여 새 CSV 파일로 저장합니다.
    """
    print("\n" + "="*70)
    print(f"PART 3: CSV 파일 동기화 (필터링) 시작")
    print(f"         (소스: '{original_csv}')")
    print(f"         (대상: '{final_csv}')")
    print("="*70)
    
    try:
        df = pd.read_csv(original_csv)
    except Exception as e:
        print(f"❌ 오류: 원본 CSV 파일을 읽을 수 없습니다: {e}")
        return False
        
    rows_before = len(df)
    
    # [핵심] 'video_id' 컬럼의 값이 master_ids (Set)에 포함된 행만 남김
    df_final = df[df['video_id'].isin(master_ids)].reset_index(drop=True)
    
    rows_after = len(df_final)
    
    # 새 파일로 저장
    try:
        df_final.to_csv(final_csv, index=False, encoding='utf-8-sig')
        
        print(f"✓ CSV 파일 동기화 완료.")
        print(f"  - 원본 행: {rows_before} 개")
        print(f"  - 필터링된 최종 행: {rows_after} 개")
        print(f"  - 최종 파일: '{final_csv}'")
        
        if rows_after != len(master_ids):
            print(f"  ⚠️ 경고: CSV 최종 행({rows_after})이 기준 ID({len(master_ids)})와 다릅니다!")
            print("         CSV 파일에 누락된 ID가 있는지 확인이 필요합니다.")
            
        return True
        
    except Exception as e:
        print(f"❌ 오류: 최종 CSV 파일 저장 실패: {e}")
        return False


if __name__ == "__main__":
    
    start_time = time.time()
    
    # 1. 기준 ID 목록 로드
    master_id_list = get_master_id_list(PNG_SOURCE_DIR)
    
    if master_id_list is not None:
        
        # 2. NPY 파일 동기화 (복사)
        sync_npy_files(
            master_ids=master_id_list,
            original_dir=ORIGINAL_NPY_DIR,
            final_dir=FINAL_NPY_DIR
        )
        
        # 3. CSV 파일 동기화 (필터링)
        sync_csv_file(
            master_ids=master_id_list,
            original_csv=ORIGINAL_CSV_FILE,
            final_csv=FINAL_CSV_FILE
        )

    end_time = time.time()
    print("\n" + "="*70)
    print("🎉 모든 동기화 작업 완료!")
    print(f"  - 최종 CSV: {FINAL_CSV_FILE}")
    print(f"  - 최종 NPY 폴더: {FINAL_NPY_DIR}")
    print(f"  - (PNG 폴더는 '{PNG_SOURCE_DIR}'를 그대로 사용하시면 됩니다)")
    print(f"  - 총 소요 시간: {end_time - start_time:.2f}초")
    print("="*70)