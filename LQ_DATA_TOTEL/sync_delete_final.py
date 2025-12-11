import os
import glob
import pandas as pd

def main():
    print("🧹 [3단계] 실제 PNG 파일이 없는 CSV 행 삭제 도구")
    print("-" * 60)

    # 1. CSV 파일 로드
    current_dir = os.getcwd()
    target_csv_name = "merged_result.csv"
    csv_path = os.path.join(current_dir, target_csv_name)

    if not os.path.exists(csv_path):
        print(f"❌ '{target_csv_name}' 파일이 없습니다. 1단계(통합)를 먼저 진행해주세요.")
        return

    try:
        print(f"📖 CSV 파일 로드 중: {target_csv_name}")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    if 'video_id' not in df.columns:
        print("❌ CSV 파일에 'video_id' 칼럼이 없습니다.")
        return

    print(f"📊 현재 CSV 데이터 개수: {len(df)}행")
    print("-" * 60)

    # 2. 기준이 될 PNG 폴더 경로 입력
    while True:
        target_dir_input = input("📂 기준이 될(실제 파일이 있는) PNG 폴더 경로를 입력하세요: ").strip()
        target_dir_input = target_dir_input.replace('"', '').replace("'", "")
        
        if os.path.isdir(target_dir_input):
            png_target_dir = target_dir_input
            print(f"✅ 대상 폴더 확인: {png_target_dir}")
            break
        else:
            print("❌ 유효하지 않은 폴더입니다. 다시 입력해주세요.")

    # 3. PNG 파일 목록 확보 (이게 기준이 됨)
    print("\n🔍 PNG 파일 스캔 중...")
    png_files = glob.glob(os.path.join(png_target_dir, "*.png"))
    
    if not png_files:
        print("⚠️ 해당 폴더에 PNG 파일이 하나도 없습니다. 작업을 중단합니다.")
        return

    # PNG 파일명(확장자 제외)을 추출하여 집합(Set)으로 저장 -> 검색 속도 획기적 증가
    # 예: /path/to/abc.png -> 'abc'
    existing_png_ids = set()
    for f in png_files:
        filename = os.path.basename(f)
        file_id = os.path.splitext(filename)[0]
        existing_png_ids.add(file_id)

    print(f"✅ 실제 존재하는 이미지 ID 개수: {len(existing_png_ids)}개")

    # 4. CSV 필터링 시뮬레이션
    # 로직: CSV의 'video_id'가 existing_png_ids 안에 있는 경우만 남김
    
    # 남길 데이터 (동기화 성공)
    df_synced = df[df['video_id'].astype(str).isin(existing_png_ids)]
    
    # 삭제될 데이터 (동기화 실패)
    deleted_count = len(df) - len(df_synced)

    print("-" * 60)
    print(f"📊 분석 결과 리포트")
    print(f"   - 📄 CSV 전체 데이터 : {len(df)}행")
    print(f"   - 🖼️ 실제 파일과 매칭됨 (유지) : {len(df_synced)}행")
    print(f"   - 🗑️ 매칭되는 파일 없음 (삭제 대상) : {deleted_count}행")
    print("-" * 60)

    if deleted_count == 0:
        print("✨ CSV 데이터가 이미 완벽하게 동기화되어 있습니다. 삭제할 행이 없습니다.")
        return

    # 5. 사용자 확인 및 저장
    while True:
        user_input = input(f"🔥 매칭되지 않는 CSV 데이터 {deleted_count}건을 삭제하고 새로 저장하시겠습니까? (y/n): ").strip().lower()
        
        if user_input == 'y':
            save_filename = "final_synced_data.csv"
            save_path = os.path.join(current_dir, save_filename)
            
            print(f"\n💾 '{save_filename}' 파일로 저장 중...")
            try:
                df_synced.to_csv(save_path, index=False, encoding='utf-8-sig')
                print(f"✅ 저장 완료! 작업이 끝났습니다.")
                print(f"👉 생성된 파일: {save_path}")
            except Exception as e:
                print(f"❌ 파일 저장 실패: {e}")
            break
        
        elif user_input == 'n':
            print("\n🛡️ 작업을 취소합니다. CSV 파일은 변경되지 않았습니다.")
            break
        
        else:
            print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")

if __name__ == "__main__":
    main()