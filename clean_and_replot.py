import os
import pandas as pd
import sys

# ==========================================
# [설정 영역] 아래 경로를 실제 환경에 맞게 수정하세요.
# ==========================================
CSV_FILE_PATH = 'master_summary_v11_cleaned_final.csv'      # video_id가 포함된 CSV 파일 경로
NPY_FOLDER_PATH = './2_npy_timeseries'  # NPY 파일들이 저장된 폴더 경로
ID_COLUMN_NAME = 'video_id'     # CSV 파일 내의 ID 컬럼 헤더 이름
# ==========================================

def delete_unmatched_files():
    # 1. CSV 파일이 존재하는지 확인
    if not os.path.exists(CSV_FILE_PATH):
        print(f"오류: CSV 파일을 찾을 수 없습니다. ({CSV_FILE_PATH})")
        return

    # 2. NPY 폴더가 존재하는지 확인
    if not os.path.exists(NPY_FOLDER_PATH):
        print(f"오류: NPY 폴더를 찾을 수 없습니다. ({NPY_FOLDER_PATH})")
        return

    try:
        # 3. CSV 파일 로드 및 ID 리스트 추출
        print("CSV 파일을 읽는 중...")
        df = pd.read_csv(CSV_FILE_PATH)
        
        # 컬럼 확인
        if ID_COLUMN_NAME not in df.columns:
            print(f"오류: CSV 파일 내에 '{ID_COLUMN_NAME}' 컬럼이 없습니다.")
            return

        # 비교를 위해 ID를 문자열(String) 집합(Set)으로 변환 (검색 속도 향상 및 타입 일치)
        valid_ids = set(df[ID_COLUMN_NAME].astype(str))
        print(f"총 {len(valid_ids)}개의 유효한 video_id를 확인했습니다.")

        # 4. 폴더 내 파일 순회 및 삭제 로직
        deleted_count = 0
        processed_count = 0
        
        # 폴더 내 파일 리스트 가져오기
        files = os.listdir(NPY_FOLDER_PATH)
        
        print("\n파일 검사 및 정리를 시작합니다...")
        
        for filename in files:
            # .npy 파일만 처리
            if filename.endswith('.npy'):
                processed_count += 1
                
                # 확장자를 제외한 순수 파일명 추출 (예: 'video_123.npy' -> 'video_123')
                file_stem = os.path.splitext(filename)[0]
                
                # CSV 목록에 없는 경우 삭제
                if file_stem not in valid_ids:
                    file_path = os.path.join(NPY_FOLDER_PATH, filename)
                    try:
                        os.remove(file_path)
                        print(f"[삭제됨] {filename} (CSV에 존재하지 않음)")
                        deleted_count += 1
                    except Exception as e:
                        print(f"[삭제 실패] {filename}: {e}")

        # 5. 결과 요약
        print("-" * 30)
        print(f"작업 완료.")
        print(f"검사한 NPY 파일 수: {processed_count}개")
        print(f"삭제한 파일 수: {deleted_count}개")
        print(f"유지된 파일 수: {processed_count - deleted_count}개")

    except Exception as e:
        print(f"스크립트 실행 중 예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    # 실수 방지를 위한 확인 절차 (필요 없다면 주석 처리 가능)
    check = input("이 작업은 파일을 영구적으로 삭제합니다. 계속하시겠습니까? (y/n): ")
    if check.lower() == 'y':
        delete_unmatched_files()
    else:
        print("작업이 취소되었습니다.")
