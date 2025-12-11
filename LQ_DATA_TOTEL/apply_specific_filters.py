import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 시각화 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.pyplot')

# --- 설정 ---
INPUT_CSV = "master_summary_no_filter.csv"          # 입력 파일
OUTPUT_CSV = "final_cleaned_interactive.csv"        # 최종 결과 저장 파일
PLOT_DIR = "plots_interactive"                      # 그래프 저장 폴더

# --- 강도(Intensity) 레벨 설정 ---
# 단계가 높을수록 Multiplier가 커져서 "덜 삭제"됩니다 (완화).
# 1단계: 1.5 (표준 통계적 이상치 기준, 엄격함)
# 5단계: 5.0 (매우 극단적인 값만 삭제, 너그러움)
LEVEL_MAP = {
    1: 1.5,
    2: 2.0,
    3: 3.0,
    4: 4.0,
    5: 6.0
}

# --- 1차: 수동 필터링 조건 (8개 고정) ---
MANUAL_FILTERS = [
    ("full_face_area_avg", "remove_ge", 10000),
    ("full_face_area_max", "remove_ge", 10000),
    ("full_face_area_min", "remove_ge", 10000),
    ("full_face_area_std", "remove_ge", 200),
    ("full_face_laplacian_mean_max", "remove_ge", 20),
    ("full_face_laplacian_mean_std", "remove_ge", 0.85),
    ("full_face_laplacian_var_avg", "remove_ge", 1000),
    ("full_face_light_intensity_change_avg", "keep_range", (-0.2, 0.2)),
]

def plot_all_distributions(df, step_name):
    """
    현재 데이터프레임의 모든 수치형 칼럼 분포를 시각화하여 저장합니다.
    """
    if df is None or df.empty:
        return

    save_dir = os.path.join(PLOT_DIR, step_name)
    os.makedirs(save_dir, exist_ok=True)

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    # 시각화가 너무 오래 걸리면 일부만 샘플링하거나 진행상황 출력
    print(f"📈 [{step_name}] 분포 시각화 생성 중... ({len(numerical_cols)}개 칼럼)")

    for col in numerical_cols:
        try:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=50)
            plt.title(f"{col}\n(Hist) - {step_name}")
            
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[col])
            plt.title(f"{col}\n(Box) - {step_name}")
            
            plt.tight_layout()
            safe_col = col.replace('/', '_').replace('\\', '_').replace(':', '')
            plt.savefig(os.path.join(save_dir, f"{safe_col}.png"))
            plt.close()
        except Exception:
            pass

def apply_manual_filters(df):
    """
    정의된 8개의 수동 조건을 적용합니다.
    """
    print("\n🔍 [1단계] 수동 필터링 (고정 조건 8개)...")
    df_filtered = df.copy()
    
    for i, (col, f_type, val) in enumerate(MANUAL_FILTERS, 1):
        if col not in df_filtered.columns:
            continue
            
        if f_type == "remove_ge":  # 이상값 제거
            mask = df_filtered[col] >= val
        elif f_type == "keep_range": # 범위 외 제거
            min_v, max_v = val
            mask = (df_filtered[col] < min_v) | (df_filtered[col] > max_v)
            
        removed = mask.sum()
        df_filtered = df_filtered[~mask]
        
        if removed > 0:
            print(f"   - 조건 {i} ({col}): {removed}개 삭제됨")
            
    print(f"   ✅ 수동 필터 완료. 남은 데이터: {len(df_filtered)}개")
    return df_filtered

def preview_auto_iqr(df, multiplier):
    """
    실제 삭제하지 않고, 삭제될 행의 개수만 시뮬레이션하여 반환합니다.
    """
    # 수치형 칼럼만 선택 (ID 등 제외)
    numeric_cols = df.select_dtypes(include=['number']).columns
    exclude_cols = ['video_id', 'label']
    target_cols = [c for c in numeric_cols if c not in exclude_cols]

    # 삭제될 인덱스를 모으는 집합 (중복 방지)
    outlier_indices = set()

    for col in target_cols:
        if df[col].nunique() <= 1: continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - (multiplier * IQR)
        upper = Q3 + (multiplier * IQR)
        
        # 이상치 인덱스 추출
        mask = (df[col] < lower) | (df[col] > upper)
        idxs = df[mask].index.tolist()
        outlier_indices.update(idxs)
        
    return len(outlier_indices), outlier_indices

def main():
    print("🚀 [대화형 데이터 정제 도구] 수동 조건 + 가변 자동 필터링")
    print("-" * 70)

    # 1. 데이터 로드
    if not os.path.exists(INPUT_CSV):
        if os.path.exists("merged_result.csv"):
            csv_path = "merged_result.csv"
        else:
            print(f"❌ 입력 파일({INPUT_CSV})을 찾을 수 없습니다.")
            return
    else:
        csv_path = INPUT_CSV

    print(f"📖 CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    original_len = len(df)
    print(f"📊 원본 데이터: {original_len}개")

    # 2. 수동 필터링 (고정)
    df_manual = apply_manual_filters(df)
    # 시각화 (수동 필터 후)
    plot_all_distributions(df_manual, "01_After_Manual_Before_Auto")

    # 3. 대화형 자동 필터링 루프
    print("\n🔍 [2단계] 자동 이상치 제거 강도 설정")
    print("   * 단계가 낮을수록 엄격합니다 (많이 삭제)")
    print("   * 단계가 높을수록 완화됩니다 (적게 삭제)")
    print("-" * 50)
    
    final_df = None
    
    while True:
        print("\n🎚️  필터링 강도를 선택하세요 (1~5):")
        print("   [1] 매우 엄격 (IQR x 1.5) - 표준 통계 기준")
        print("   [2] 엄격      (IQR x 2.0)")
        print("   [3] 보통      (IQR x 3.0)")
        print("   [4] 완화      (IQR x 4.0)")
        print("   [5] 매우 완화 (IQR x 6.0) - 극단적 이상치만 제거")
        
        user_choice = input("👉 번호 입력 (1-5): ").strip()
        
        if not user_choice.isdigit() or int(user_choice) not in LEVEL_MAP:
            print("❌ 잘못된 입력입니다. 1에서 5 사이의 숫자를 입력해주세요.")
            continue
            
        level = int(user_choice)
        multiplier = LEVEL_MAP[level]
        
        # 시뮬레이션 진행
        print(f"\n⏳ [시뮬레이션] 강도 {level}단계 (Multiplier {multiplier}) 분석 중...")
        remove_count, remove_indices = preview_auto_iqr(df_manual, multiplier)
        
        remain_count = len(df_manual) - remove_count
        percent = (remove_count / len(df_manual)) * 100
        
        print("-" * 50)
        print(f"📊 [예상 결과]")
        print(f"   - 현재 데이터 수 : {len(df_manual)}개")
        print(f"   - 삭제될 데이터  : {remove_count}개 ({percent:.2f}%)")
        print(f"   - 예상 남는 데이터: {remain_count}개")
        print("-" * 50)
        
        confirm = input("🔥 이대로 삭제를 진행하시겠습니까? (y:확인 / n:다시설정): ").strip().lower()
        
        if confirm == 'y':
            # 실제 삭제 진행
            final_df = df_manual.drop(index=remove_indices).reset_index(drop=True)
            
            # 최종 시각화
            step_name = f"02_Final_Intensity_Level_{level}"
            plot_all_distributions(final_df, step_name)
            print(f"✅ 적용 완료! '{step_name}' 폴더에 시각화 저장됨.")
            break
        
        elif confirm == 'n':
            print("🔄 강도 설정을 다시 진행합니다.")
        else:
            print("❌ 잘못된 입력입니다. 다시 설정으로 돌아갑니다.")

    # 4. 결과 저장
    print("\n" + "-" * 70)
    print(f"💾 최종 결과 저장 중...")
    try:
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ 저장 완료: {OUTPUT_CSV}")
        
        total_removed = original_len - len(final_df)
        print(f"📊 최종 리포트:")
        print(f"   - 원본: {original_len} -> 최종: {len(final_df)}")
        print(f"   - 총 삭제된 행: {total_removed}개")
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

if __name__ == "__main__":
    main()