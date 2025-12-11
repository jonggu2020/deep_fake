import os
import glob
import pandas as pd
import time

def main():
    # 1. 현재 작업 경로 설정
    current_dir = os.getcwd()
    print(f"📂 현재 작업 경로: {current_dir}")
    
    # 2. CSV 파일 리스트 가져오기
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    
    # 만약 이전에 생성된 통합 파일(merged_result.csv)이 있다면 리스트에서 제외 (중복 방지)
    output_filename = "merged_result.csv"
    output_path = os.path.join(current_dir, output_filename)
    
    if output_path in csv_files:
        csv_files.remove(output_path)
        print(f"ℹ️  기존 통합 파일({output_filename})은 병합 대상에서 제외합니다.")

    if not csv_files:
        print("❌ 현재 경로에 병합할 CSV 파일이 없습니다.")
        return

    print(f"🔍 발견된 CSV 파일 개수: {len(csv_files)}개")
    print("-" * 50)

    # 3. 데이터프레임 읽기 및 리스트에 추가
    df_list = []
    for file in csv_files:
        try:
            # 파일명 출력
            file_name = os.path.basename(file)
            print(f"reading... {file_name}")
            
            # csv 읽기 (인코딩 에러 방지 차원에서 utf-8 시도)
            df = pd.read_csv(file, encoding='utf-8')
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ {file_name} 로드 실패: {e}")

    # 4. 데이터 통합
    if df_list:
        print("-" * 50)
        print("🔄 데이터 병합 중...")
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # 5. 결과 저장
        try:
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 통합 완료!")
            print(f"💾 저장된 파일명: {output_filename}")
            print(f"📊 총 데이터 행 수: {len(merged_df)}개")
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")
    else:
        print("❌ 병합할 유효한 데이터가 없습니다.")

if __name__ == "__main__":
    main()