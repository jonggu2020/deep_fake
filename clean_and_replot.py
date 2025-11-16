import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

# Matplotlib에서 수많은 플롯을 생성할 때 발생하는 경고를 억제합니다.
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.pyplot')

# --- 1. 사용자 설정 ---

# ⚠️ [수정필요] 원본 통합 CSV 파일 경로
MERGED_CSV_FILE = "./master_summary_v1_standard.csv" 

# ⚠️ [출력 1] 정제된 데이터가 저장될 CSV 파일 이름 (v11로 변경)
OUTPUT_CLEANED_CSV = "./master_summary_v11_cleaned_final.csv"

# ⚠️ [출력 2] 정제된 데이터의 분포도 그래프가 저장될 *새* 폴더 (v9로 변경)
CLEANED_PLOT_DIR = "./cleaned_distribution_plots_v9"

# ---

def filter_outliers_v9(csv_file, output_file):
    """
    (PART 1) CSV의 분포도를 기준으로 이상치를 제거합니다. (v9: 총 27개 조건 반영)
    제거 후, 정제된 DataFrame을 반환합니다.
    """
    print("="*70)
    print(f"PART 1: 이상치 필터링 시작 (v9 - 총 27개 조건)")
    print("="*70)
    print(f"'{csv_file}' 파일을 불러오는 중...")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None 

    total_rows_before = len(df)
    print(f"✓ 원본 데이터: {total_rows_before} 행")
    
    # --- 이상치 제거 기준 설정 (총 27개) ---
    print("\n--- 적용될 필터 기준 (총 27개) ---")
    
    # (컬럼명, 연산자, 임계값) 튜플의 리스트로 기준 정의
    # 연산자: 'quantile_lt' (보다 작음), 'quantile_gt' (보다 큼), 'lt' (보다 작음), 'gt' (보다 큼), 'between' (양쪽 극단값)
    
    filter_criteria = [
        # 1-3 (full_face)
        ('full_face_laplacian_mean_avg', 'quantile_lt', 0.05),
        ('full_face_light_intensity_mean_avg', 'between', (10.0, 250.0)), # (10.0 미만 또는 250.0 초과)
        ('full_face_laplacian_var_max', 'quantile_gt', 0.95),
        
        # 4-6 (light_intensity_change)
        ('left_eye_light_intensity_change_max', 'gt', 40.0),
        ('full_face_light_intensity_change_max', 'gt', 5.0),
        ('full_face_light_intensity_change_min', 'lt', -5.0),
        
        # 7-9 (jawline, nose)
        ('jawline_light_intensity_change_max', 'gt', 5.0),
        ('nose_light_intensity_change_max', 'gt', 4.0),
        ('nose_light_intensity_mean_std', 'gt', 2.0),
        
        # 10-13 (nose laplacian, mouth)
        ('nose_laplacian_var_min', 'gt', 70.0),
        ('nose_laplacian_var_std', 'gt', 20.0),
        ('nose_laplacian_var_avg', 'gt', 100.0),
        ('mouth_light_intensity_change_max', 'gt', 20.0),

        # 14-17 (left_eye)
        ('left_eye_laplacian_var_min', 'gt', 620.0),
        ('left_eye_laplacian_var_std', 'gt', 250.0),
        ('left_eye_light_intensity_change_min', 'lt', -20.0),
        ('left_eye_light_intensity_change_std', 'gt', 7.5),
        
        # 18-20 (right_eye light)
        ('right_eye_light_intensity_mean_std', 'gt', 10.0),
        ('right_eye_light_intensity_change_std', 'gt', 7.5),
        ('right_eye_light_intensity_change_min', 'lt', -25.0),
        
        # 21-23 (mouth)
        ('mouth_laplacian_var_min', 'gt', 110.0),
        ('mouth_laplacian_var_std', 'gt', 60.0),
        ('mouth_light_intensity_change_min', 'lt', -17.0),
        
        # 24-26 (right_eye)
        ('right_eye_laplacian_var_min', 'gt', 620.0),
        ('right_eye_laplacian_var_std', 'gt', 270.0),
        ('right_eye_light_intensity_change_max', 'gt', 24.0),
        
        # 27 (left_eye_area)
        ('left_eye_area_std', 'gt', 280.0)
    ]

    df_filtered = df.copy()
    all_indices_to_drop = set()
    filter_candidate_counts = []

    # --- 필터링 적용 ---
    for i, (col, op, value) in enumerate(filter_criteria, 1):
        if col not in df_filtered.columns:
            print(f"  [필터 {i}] 경고: '{col}' 컬럼이 없습니다. 건너뜁니다.")
            filter_candidate_counts.append((f"필터 {i} ({col})", 0))
            continue
            
        indices_to_drop = set()
        
        try:
            if op == 'quantile_lt':
                threshold = df_filtered[col].quantile(value)
                indices_to_drop = set(df_filtered[df_filtered[col] < threshold].index)
                print(f"  [{i:2d}] {col} < {threshold:.2f} (하위 {value*100}%)")
            
            elif op == 'quantile_gt':
                threshold = df_filtered[col].quantile(value)
                indices_to_drop = set(df_filtered[df_filtered[col] > threshold].index)
                print(f"  [{i:2d}] {col} > {threshold:.2f} (상위 {(1-value)*100:.0f}%)")
            
            elif op == 'lt':
                threshold = value
                indices_to_drop = set(df_filtered[df_filtered[col] < threshold].index)
                print(f"  [{i:2d}] {col} < {threshold}")
            
            elif op == 'gt':
                threshold = value
                indices_to_drop = set(df_filtered[df_filtered[col] > threshold].index)
                print(f"  [{i:2d}] {col} > {threshold}")
            
            elif op == 'between':
                low, high = value
                indices_to_drop = set(df_filtered[
                    (df_filtered[col] < low) | (df_filtered[col] > high)
                ].index)
                print(f"  [{i:2d}] {col} < {low} 또는 > {high}")
                
            filter_candidate_counts.append((f"필터 {i} ({col})", len(indices_to_drop)))
            all_indices_to_drop.update(indices_to_drop)
            
        except Exception as e:
            print(f"  [필터 {i}] 오류: '{col}' 처리 중 오류 발생: {e}")
            filter_candidate_counts.append((f"필터 {i} ({col})", 0))

                      
    total_to_drop = len(all_indices_to_drop)

    if total_to_drop > 0:
        print(f"\n--- 필터링 요약 ---")
        
        # 각 필터별 제거 후보 개수 출력 (중복 포함)
        for filter_name, count in filter_candidate_counts:
            print(f"  - {filter_name} 후보: {count} 개")
            
        print(f"  ▶ 제거 대상 총합 (중복 제거): {total_to_drop} 개")
        
        # '제거 대상'이 *아닌* 인덱스만 선택
        df_cleaned = df_filtered[~df_filtered.index.isin(all_indices_to_drop)].reset_index(drop=True)
        
        total_rows_after = len(df_cleaned)
        percent_removed = (total_to_drop / total_rows_before) * 100

        print(f"✓ 필터링 완료!")
        print(f"  - 원본 행: {total_rows_before}")
        print(f"  - 제거된 행: {total_to_drop} ({percent_removed:.2f}%)")
        print(f"  - 남은 행: {total_rows_after}")
    
    else:
        print("\n✓ 제거할 이상치가 발견되지 않았습니다. 원본 데이터를 그대로 사용합니다.")
        df_cleaned = df_filtered

    # 4. 정제된 데이터를 새 CSV 파일로 저장
    try:
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 성공! 정제된 파일이 '{output_file}'(으)로 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 최종 파일 저장 실패: {e}")

    # 5. 다음 단계를 위해 정제된 DataFrame을 반환
    return df_cleaned


def generate_plots(df, plot_dir):
    """
    (PART 2) 전달받은 DataFrame의 분포도를 생성하여 'plot_dir'에 저장합니다.
    """
    print("\n" + "="*70)
    print(f"PART 2: 정제된 데이터 분포도 생성 시작")
    print(f"         (저장 위치: {plot_dir})")
    print("="*70)

    if df is None or df.empty:
        print("❌ 분포도를 그릴 데이터가 없습니다 (DataFrame이 비어있음).")
        return

    os.makedirs(plot_dir, exist_ok=True)
    
    # --- 컬럼 유형 분리 ---
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    print(f"  - 숫자형 컬럼 (히스토그램 생성 대상): {len(numerical_cols)}개")
    print(f"  - 범주형 컬럼 (카운트 플롯 생성 대상): {len(categorical_cols)}개")

    # --- 숫자형 컬럼 (히스토그램) 생성 ---
    print("\n... 1. 숫자형 데이터 히스토그램 생성 중 ...")
    numeric_plot_dir = os.path.join(plot_dir, "numerical_histograms")
    os.makedirs(numeric_plot_dir, exist_ok=True)
    
    total_num = len(numerical_cols)
    start_numeric = time.time()
    
    for i, col in enumerate(numerical_cols, 1):
        try:
            data_series = df[col].dropna()
            if data_series.empty:
                continue
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data_series, kde=True, bins=50) 
            mean_val = data_series.mean()
            std_val = data_series.std()
            plt.title(f"Histogram: {col}\n(Mean: {mean_val:.2f}, Std: {std_val:.2f})")
            plt.xlabel(col)
            plt.ylabel("Frequency (빈도)")
            plt.tight_layout()
            safe_col_name = col.replace(os.path.sep, '_').replace(':', '_')
            output_path = os.path.join(numeric_plot_dir, f"hist_{safe_col_name}.png")
            plt.savefig(output_path, dpi=90)
            plt.close()
        except Exception as e:
            print(f"    ❌ '{col}' 플롯 생성 실패: {e}")
            
    numeric_time = time.time() - start_numeric
    print(f"✓ 숫자형 컬럼 처리 완료 (소요 시간: {numeric_time:.2f}초)")

    # --- 범주형 컬럼 (카운트 플롯) 생성 ---
    print("\n... 2. 범주형 데이터 카운트 플롯 생성 중 ...")
    categorical_plot_dir = os.path.join(plot_dir, "categorical_countplots")
    os.makedirs(categorical_plot_dir, exist_ok=True)

    total_cat = len(categorical_cols)
    start_categorical = time.time()
    
    for i, col in enumerate(categorical_cols, 1):
        try:
            unique_count = df[col].nunique()
            if unique_count > 100 or unique_count == 0 or df[col].isnull().all():
                continue
        
            chart_width = max(10, unique_count * 0.5)
            plt.figure(figsize=(chart_width, 7))
            order = df[col].value_counts().index
            sns.countplot(data=df, x=col, order=order)
            plt.title(f"Count Plot: {col} (Unique Values: {unique_count})")
            plt.xlabel(col)
            plt.ylabel("Count (개수)")
            if unique_count > 5:
                 plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            safe_col_name = col.replace(os.path.sep, '_').replace(':', '_')
            output_path = os.path.join(categorical_plot_dir, f"count_{safe_col_name}.png")
            plt.savefig(output_path, dpi=90)
            plt.close()
        except Exception as e:
            print(f"    ❌ '{col}' 플롯 생성 실패: {e}")
            
    categorical_time = time.time() - start_categorical
    print(f"✓ 범주형 컬럼 처리 완료 (소요 시간: {categorical_time:.2f}초)")


if __name__ == "__main__":
    
    print("--- [시작] 이상치 제거 및 분포도 재생성 (v9 - 총 27개 조건) ---")
    
    # 1단계: 필터링 실행 (v9 함수 호출)
    df_cleaned = filter_outliers_v9(
        csv_file=MERGED_CSV_FILE, 
        output_file=OUTPUT_CLEANED_CSV
    )
    
    # 2단계: 1단계에서 반환된 'df_cleaned'를 사용하여 분포도 생성
    if df_cleaned is not None:
        generate_plots(
            df=df_cleaned, 
            plot_dir=CLEANED_PLOT_DIR
        )
    else:
        print("\n❌ 1단계(필터링)에서 오류가 발생하여 2단계(플롯 생성)를 건너뜁니다.")

    print("\n" + "="*70)
    print("🎉 모든 작업 완료!")
    print(f"  - 정제된 CSV: {OUTPUT_CLEANED_CSV}")
    print(f"  - 새 분포도: {CLEANED_PLOT_DIR}")
    print("="*70)