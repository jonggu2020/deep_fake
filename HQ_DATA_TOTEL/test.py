import pandas as pd
import glob
import os
import time

# 1. 팀원들의 CSV 파일이 모여있는 폴더 경로
# (예: 이 스크립트와 같은 위치에 'all_csvs'라는 폴더를 만드세요)
CSV_SOURCE_DIR = "./data" 

# 2. 통합된 마스터 CSV 파일을 저장할 경로와 이름
OUTPUT_MASTER_CSV = "./master_summary_v1_standard.csv"

def merge_csv_files(source_dir, output_file):
    """
    (표준 방식)
    폴더 내 모든 CSV를 메모리로 읽어들인 후, pandas.concat으로 병합합니다.
    """
    start_time = time.time()
    print(f"'{source_dir}' 폴더에서 CSV 파일 검색 중...")
    
    # source_dir 경로와 그 하위 폴더까지 모든 .csv 파일을 검색합니다.
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ 검색된 CSV 파일이 없습니다. 경로를 확인하세요.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    all_dataframes = []
    for i, f in enumerate(csv_files, 1):
        try:
            print(f"  [{i}/{len(csv_files)}] 읽는 중: {f}")
            df = pd.read_csv(f)
            all_dataframes.append(df)
        except Exception as e:
            print(f"  ⚠️ 파일을 읽는 중 오류 발생: {f} ({e})")

    if not all_dataframes:
        print("❌ 유효한 데이터를 읽어오지 못했습니다.")
        return
        
    print(f"\n... {len(all_dataframes)}개 데이터프레임 병합 중 (pandas.concat) ...")
    
    # 리스트에 담긴 모든 데이터프레임을 위아래로(axis=0) 합칩니다.
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"✓ 총 {len(master_df)}개의 행으로 데이터가 통합되었습니다.")
    
    # (선택 사항) 'video_id' 컬럼 기준으로 중복된 행이 있는지 확인
    # 만약 중복이 있다면, 팀원 간에 동일한 비디오를 처리했을 수 있습니다.
    duplicates = master_df.duplicated(subset=['video_id']).sum()
    if duplicates > 0:
        print(f"  ⚠️ 경고: 'video_id'가 중복된 행이 {duplicates}개 있습니다.")
        # 중복 제거가 필요하다면 아래 주석 해제
        # master_df = master_df.drop_duplicates(subset=['video_id'], keep='first')
        # print(f"     -> 중복 제거 후 {len(master_df)}개 행이 남았습니다.")

    # 최종 파일로 저장
    try:
        print(f"\n... 최종 파일 저장 중: {output_file} ...")
        master_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        end_time = time.time()
        print(f"\n✅ 성공! 통합된 파일이 '{output_file}'(으)로 저장되었습니다.")
        print(f"   (총 소요 시간: {end_time - start_time:.2f}초)")
    except Exception as e:
        print(f"\n❌ 최종 파일 저장 실패: {e}")

if __name__ == "__main__":
    # 스크립트 실행 시 CSV 파일이 있는 폴더가 없다면 생성
    if not os.path.exists(CSV_SOURCE_DIR):
        os.makedirs(CSV_SOURCE_DIR)
        print(f"'{CSV_SOURCE_DIR}' 폴더를 생성했습니다.")
        print("이 폴더에 팀원들의 '1_statistics_all_summary.csv' 파일들을 넣고 다시 실행해주세요.")
    else:
        merge_csv_files(CSV_SOURCE_DIR, OUTPUT_MASTER_CSV)