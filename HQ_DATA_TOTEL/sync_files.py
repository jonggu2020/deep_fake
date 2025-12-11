import pandas as pd
import os
import glob
import time

# --- 1. 사용자 설정 ---

# ⚠️ [수정필요 1]
# 최종 정제된 "기준" CSV 파일 경로
# (v9 스크립트를 실행했다면 'master_summary_v11_cleaned_final.csv'가 맞습니다)
CLEANED_CSV_FILE = "./master_summary_v11_cleaned_final.csv" 

# ⚠️ [수정필요 2]
# 정리할 NPY 파일이 있는 폴더 경로
NPY_DIR = "./2_npy_timeseries"

# ⚠️ [수정필요 3]
# 정리할 PNG 파일이 있는 폴더 경로
PNG_DIR = "./3_audio_spectrograms"

# ---

def load_valid_ids(csv_file):
    """
    기준이 되는 CSV 파일을 로드하여, 유효한 video_id 목록을
    빠른 조회를 위해 'Set' 자료구조로 반환합니다.
    """
    print("="*70)
    print(f"PART 1: 기준 CSV 파일 로드")
    print("="*70)
    print(f"'{csv_file}' 파일에서 유효한 ID 목록을 불러오는 중...")
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'video_id' not in df.columns:
            print(f"❌ 오류: CSV 파일에 'video_id' 컬럼이 없습니다.")
            return None
            
        # 리스트가 아닌 Set(집합)으로 만들어야 조회 속도가 수백 배 빨라집니다.
        valid_ids_set = set(df['video_id'])
        
        print(f"✓ {len(valid_ids_set)}개의 고유한(Unique) 유효 ID를 로드했습니다.")
        return valid_ids_set
        
    except FileNotFoundError:
        print(f"❌ 오류: 기준 CSV 파일을 찾을 수 없습니다: {csv_file}")
        return None
    except Exception as e:
        print(f"❌ 오류: CSV 파일 로드 중 문제 발생: {e}")
        return None


def sync_folder(folder_path, file_extension, valid_ids):
    """
    (PART 2 & 3)
    지정된 폴더의 파일들을 'valid_ids' 목록과 비교하여,
    목록에 없는 파일은 삭제합니다.
    """
    print("\n" + "="*70)
    print(f"PART {2 if file_extension == 'npy' else 3}: '{folder_path}' 폴더 동기화 (.{file_extension} 파일)")
    print("="*70)

    if not os.path.isdir(folder_path):
        print(f"❌ 오류: 폴더를 찾을 수 없습니다: {folder_path}")
        print("   경로를 확인하세요. 이 폴더를 건너뜁니다.")
        return 0, 0, 0 # (total, kept, deleted)

    # glob을 사용하여 해당 확장자의 모든 파일 경로를 가져옵니다.
    file_paths = glob.glob(os.path.join(folder_path, f"*.{file_extension}"))
    
    total_files = len(file_paths)
    deleted_count = 0
    kept_count = 0

    if total_files == 0:
        print(f"✓ 폴더에 '*.{file_extension}' 파일이 없습니다. (처리할 것 없음)")
        return 0, 0, 0

    print(f"총 {total_files}개의 '*.{file_extension}' 파일을 발견했습니다.")
    print(f"'{len(valid_ids)}'개의 유효 ID와 비교하여 삭제 작업을 시작합니다...")

    for file_path in file_paths:
        try:
            # 파일명에서 확장자를 제거하여 'base_name' (video_id)을 추출합니다.
            # 예: "2_npy_timeseries/video1_speech_early.npy" -> "video1_speech_early"
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # [핵심 로직]
            # 추출한 base_name이 유효한 ID 목록(Set)에 있는지 확인
            if base_name in valid_ids:
                # 목록에 있으므로, 파일을 유지합니다.
                kept_count += 1
            else:
                # 목록에 없으므로, 파일을 삭제합니다.
                os.remove(file_path)
                deleted_count += 1
                
        except Exception as e:
            print(f"  ⚠️ 파일 처리 중 오류 발생 (파일: {file_path}): {e}")

    print(f"✓ '{folder_path}' 폴더 동기화 완료.")
    print(f"  - 총 파일: {total_files}")
    print(f"  - 유지된 파일: {kept_count}")
    print(f"  - 삭체된 파일: {deleted_count}")
    
    return total_files, kept_count, deleted_count


if __name__ == "__main__":
    
    start_time = time.time()
    
    # 1단계: 유효한 ID 목록 로드
    valid_id_set = load_valid_ids(CLEANED_CSV_FILE)
    
    if valid_id_set is not None:
        # 2단계: NPY 폴더 동기화
        npy_total, npy_kept, npy_deleted = sync_folder(
            folder_path=NPY_DIR, 
            file_extension="npy", 
            valid_ids=valid_id_set
        )
        
        # 3단계: PNG 폴더 동기화
        png_total, png_kept, png_deleted = sync_folder(
            folder_path=PNG_DIR,
            file_extension="png",
            valid_ids=valid_id_set
        )
        
        # --- 최종 요약 ---
        end_time = time.time()
        print("\n" + "="*70)
        print("🎉 모든 파일 동기화 작업 완료!")
        print("="*70)
        print(f"  ⏱️ 총 소요 시간: {end_time - start_time:.2f}초")
        print(f"\n  📂 NPY 폴더 ('{NPY_DIR}')")
        print(f"    - 총 {npy_total}개 중 {npy_deleted}개 삭제, {npy_kept}개 유지됨.")
        print(f"\n  📂 PNG 폴더 ('{PNG_DIR}')")
        print(f"    - 총 {png_total}개 중 {png_deleted}개 삭제, {png_kept}개 유지됨.")
        print("\n  ℹ️ CSV 파일의 유효 ID 개수와 '유지된 파일' 개수가 일치해야 합니다.")
        print(f"    - CSV 유효 ID: {len(valid_id_set)} 개")
        print(f"    - NPY 유지됨: {npy_kept} 개")
        print(f"    - PNG 유지됨: {png_kept} 개")
        print("="*70)
    
    else:
        print("\n❌ 기준 CSV 파일 로드에 실패하여 동기화 작업을 시작할 수 없습니다.")