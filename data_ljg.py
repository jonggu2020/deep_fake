import pandas as pd
import numpy as np
import os
import tqdm  # 진행상황 표시를 위해 추가 (없으면 pip install tqdm 필요, 없어도 돌아가게 처리함)

def load_multiple_datasets_as_dict(hq_file_paths):
    """
    [업그레이드] 여러 개의 고화질 CSV 파일들을 리스트로 받아
    하나의 거대한 Dictionary로 통합합니다.
    파일이 2개든 10개든 상관없이 모든 데이터를 검색 가능하게 만듭니다.
    """
    full_data_dict = {}
    total_loaded = 0
    
    print(f"[데이터 로딩] 고화질 원본 데이터 로드 시작...")
    
    for path in hq_file_paths:
        if not os.path.exists(path):
            print(f"   [경고] 파일을 찾을 수 없습니다: {path}")
            continue
            
        print(f"   -> 읽는 중: {os.path.basename(path)}")
        try:
            # 헤더가 없다고 가정하고 모두 읽되, DtypeWarning을 방지하기 위해 low_memory=False 사용
            # 모든 데이터를 일단 문자열이나 객체로 처리하여 로드합니다.
            df = pd.read_csv(path, header=None, low_memory=False)
            
            # ID 중복 방지를 위해 확인
            # Key: 첫 번째 컬럼(ID), Value: 전체 행 데이터(list)
            df.set_index(0, inplace=True, drop=False) 
            data_dict = df.T.to_dict('list')
            
            full_data_dict.update(data_dict) # 기존 사전에 추가 (Merge)
            total_loaded += len(df)
            
        except Exception as e:
            print(f"   [에러] 파일 로드 실패 ({path}): {e}")

    print(f"   [완료] 총 {len(hq_file_paths)}개 파일에서 {len(full_data_dict)}개의 고유 ID를 확보했습니다.")
    return full_data_dict

def create_matched_clean_dataset(reference_csv_path, hq_dictionary, output_path, scale_ratio):
    """
    저화질 파일(Reference)의 ID 순서에 맞춰, 
    미리 로딩된 고화질 딕셔너리(hq_dictionary)에서 값을 찾아 변환합니다.
    """
    print(f"\n[변환 시작] Target: {os.path.basename(output_path)}")
    print(f"- 기준 파일(Ref): {os.path.basename(reference_csv_path)}")

    # 기준 파일 읽기 (경고 방지 옵션 추가)
    try:
        ref_df = pd.read_csv(reference_csv_path, header=None, low_memory=False)
    except Exception as e:
        print(f"[에러] 기준 파일 읽기 실패: {e}")
        return

    converted_rows = []
    missing_count = 0
    
    # 진행률 표시 설정
    iterator = range(len(ref_df))
    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(ref_df)), desc="   Processing")
    except ImportError:
        pass

    for i in iterator:
        target_id = ref_df.iloc[i, 0] # ID 추출
        
        # 통합된 고화질 딕셔너리에서 검색
        if target_id in hq_dictionary:
            hq_row = hq_dictionary[target_id]
            
            meta_part = hq_row[:2] # ID, Label (앞부분 2개 컬럼)
            numeric_part = hq_row[2:] # 좌표값들 (뒷부분 나머지)
            
            # [수정된 부분] 안전한 스케일링 로직
            # 헤더(텍스트)가 들어오면 에러가 나므로, try-except로 처리합니다.
            try:
                # 모든 값이 숫자로 변환 가능한 경우에만 나눗셈 수행
                scaled_numeric = [float(x) / scale_ratio for x in numeric_part]
            except ValueError:
                # 에러 발생(즉, 'left_eye...' 같은 문자열 헤더인 경우) -> 변환 없이 그대로 사용
                # 이렇게 하면 결과 파일에도 헤더가 자동으로 포함됩니다.
                scaled_numeric = numeric_part
            
            new_row = meta_part + scaled_numeric
            converted_rows.append(new_row)
        else:
            missing_count += 1

    # 저장
    if converted_rows:
        result_df = pd.DataFrame(converted_rows)
        result_df.to_csv(output_path, header=False, index=False)
        print(f"   [성공] 저장 완료: {output_path}")
        if missing_count > 0:
            print(f"   [주의] 매칭 실패(누락): {missing_count}건")
    else:
        print("   [실패] 생성된 데이터가 없습니다.")

# -------------------------------------------------------
# [사용자 설정 구역] - 여기만 수정하세요
# -------------------------------------------------------
if __name__ == "__main__":
    # 1. 비율 설정
    HQ_WIDTH = 1920
    LQ_WIDTH = 426   # 실제 변환한 저화질 영상의 가로 크기 (예: 426, 320 등)
    SCALE_RATIO = HQ_WIDTH / LQ_WIDTH

    # 2. [고화질] 원본 파일 리스트 (여기에 2개를 모두 적으세요)
    # 실제 사용하시는 파일명으로 수정되어 있습니다.
    hq_files_list = [
        "HQ_CNN.csv",           # 첫 번째 고화질 파일
        "HQ_LSTM_XGBOOST.csv"   # 두 번째 고화질 파일
    ]

    # 3. [저화질/모델별] 작업 리스트
    # "ref": 현재 모델 학습에 쓰고 있는 파일 (ID 확인용)
    # "out": 새로 만들어서 저장할 파일 이름
    tasks = [
        {
            "ref": "LQ_CNN.csv", 
            "out": "model_A_train_CLEAN.csv"
        },
        # 두 번째 모델 파일명도 정확하다면 아래 주석을 풀고 파일명을 수정해서 쓰세요
        {
             "ref": "LQ_LSTM_XGBOOST.csv",
             "out": "model_B_train_CLEAN.csv"
         }
    ]

    # --- 실행 로직 (수정 불필요) ---
    # 1단계: 고화질 데이터 메모리에 통합 로드
    full_hq_dict = load_multiple_datasets_as_dict(hq_files_list)

    # 2단계: 각 작업별로 매칭 및 변환 수행
    if full_hq_dict:
        for task in tasks:
            if os.path.exists(task["ref"]):
                create_matched_clean_dataset(
                    task["ref"], 
                    full_hq_dict, 
                    task["out"], 
                    SCALE_RATIO
                )
            else:
                print(f"\n[Skip] 기준 파일을 찾을 수 없음: {task['ref']}")