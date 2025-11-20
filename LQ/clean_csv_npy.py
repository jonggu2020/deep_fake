import os
import pandas as pd

def synchronize_npy_with_confirmation(csv_path, folder_path, id_column='video_id'):
    """
    CSV의 ID 컬럼에 없는 .npy 파일들을 식별하고, 
    사용자 동의(y/n)를 얻은 후 삭제를 진행합니다.
    """
    
    # 1. CSV 파일 로드 및 유효한 ID 추출
    try:
        print(f"[준비] CSV 파일을 읽는 중... ({csv_path})")
        df = pd.read_csv(csv_path)
        # 비교를 위해 문자열로 변환 및 set 저장 (검색 속도 최적화)
        valid_ids = set(df[id_column].astype(str))
        print(f"[정보] CSV 로드 완료. 유지할 ID 개수: {len(valid_ids)}개")
        
    except FileNotFoundError:
        print(f"[에러] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    except KeyError:
        print(f"[에러] CSV 파일 내에 '{id_column}' 컬럼이 존재하지 않습니다.")
        return

    # 2. 폴더 확인
    if not os.path.exists(folder_path):
        print(f"[에러] NPY 폴더를 찾을 수 없습니다: {folder_path}")
        return

    # 3. 삭제 대상 식별 (실제 삭제 아님)
    files = os.listdir(folder_path)
    files_to_delete = [] # 삭제할 파일 목록을 담을 리스트
    
    print(f"\n[검사 중] '{folder_path}' 폴더 내 NPY 파일을 스캔하고 있습니다...")

    for file_name in files:
        # .npy 파일만 대상 (대소문자 구분 없이)
        if file_name.lower().endswith('.npy'):
            # 확장자 제외한 이름 추출 (예: 'video_123.npy' -> 'video_123')
            name_without_ext = os.path.splitext(file_name)[0]
            
            # 유효한 ID 목록에 없으면 삭제 리스트에 추가
            if name_without_ext not in valid_ids:
                files_to_delete.append(file_name)

    count_total = len([f for f in files if f.lower().endswith('.npy')])
    count_delete = len(files_to_delete)

    # 4. 결과 요약 및 사용자 확인
    print("-" * 60)
    print(f"검사 결과:")
    print(f" - 폴더 내 전체 NPY 파일: {count_total}개")
    print(f" - CSV에 없어 삭제될 파일: {count_delete}개")
    print("-" * 60)

    if count_delete == 0:
        print("삭제할 파일이 없습니다. 모든 파일이 CSV 데이터와 일치합니다.")
        return

    # 삭제 리스트 중 일부 예시 출력
    if count_delete > 0:
        print("삭제 예정 파일 예시 (최대 5개):")
        for f in files_to_delete[:5]:
            print(f" - {f}")
        if count_delete > 5:
            print(f" ... 외 {count_delete - 5}개")
    
    print("-" * 60)
    
    # 5. 사용자 입력 대기 (y/n)
    user_input = input(f"정말로 위 {count_delete}개의 NPY 파일을 삭제하시겠습니까? (y/n): ").strip().lower()

    if user_input == 'y':
        print("\n[작업 시작] 파일 삭제를 진행합니다...")
        deleted_count = 0
        
        for file_name in files_to_delete:
            full_path = os.path.join(folder_path, file_name)
            try:
                os.remove(full_path)
                deleted_count += 1
            except OSError as e:
                print(f"[삭제 실패] {file_name}: {e}")
        
        print(f"\n[완료] 총 {deleted_count}개의 파일이 성공적으로 삭제되었습니다.")
        
    else:
        print("\n[취소] 사용자가 작업을 취소했습니다. 파일이 삭제되지 않았습니다.")

# ========================================================
# [사용자 설정 영역] 공유해주신 파일명을 기본값으로 넣었습니다.
# ========================================================

# 1. CSV 파일 경로
target_csv_path = './final_cleaned_interactive.csv'

# 2. NPY 파일이 있는 폴더 경로
target_npy_folder = './2_npy_timeseries'

# 3. CSV 내의 ID 컬럼 이름
target_column_name = 'video_id'

# 함수 실행
if __name__ == "__main__":
    synchronize_npy_with_confirmation(target_csv_path, target_npy_folder, target_column_name)