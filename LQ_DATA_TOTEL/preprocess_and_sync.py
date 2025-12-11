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
MERGED_CSV_FILE = "./final_synced_data.csv" 

# ⚠️ [출력 1] 저장될 CSV 파일 이름 (필터링 없음_v1)
OUTPUT_CLEANED_CSV = "./master_summary_no_filter.csv"

# ⚠️ [출력 2] 분포도 그래프가 저장될 폴더 (필터링 없음_v1)
CLEANED_PLOT_DIR = "./distribution_plots_no_filter"

# ---

def process_without_filter(csv_file, output_file):
    """
    (PART 1) 조건을 모두 제거하고 원본 데이터를 그대로 로드 및 저장합니다.
    """
    print("="*70)
    print(f"PART 1: 데이터 로드 (필터링 조건 없음)")
    print("="*70)
    print(f"'{csv_file}' 파일을 불러오는 중...")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None 

    total_rows = len(df)
    print(f"✓ 원본 데이터: {total_rows} 행")
    
    # --- 필터링 로직 제거됨 ---
    print("\n>>> 모든 필터링 조건을 제거했습니다. 데이터를 삭제하지 않습니다. <<<")
    
    # 정제된 데이터(사실상 원본)를 새 CSV 파일로 저장
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 원본 그대로 파일이 '{output_file}'(으)로 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 파일 저장 실패: {e}")

    return df


def generate_plots(df, plot_dir):
    """
    (PART 2) 전달받은 DataFrame의 분포도를 생성하여 'plot_dir'에 저장합니다.
    """
    print("\n" + "="*70)
    print(f"PART 2: 전체 데이터 분포도 생성 시작")
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
            # bins를 100으로 늘려 더 정밀하게 분포를 확인하도록 함
            sns.histplot(data_series, kde=True, bins=100) 
            mean_val = data_series.mean()
            std_val = data_series.std()
            min_val = data_series.min()
            max_val = data_series.max()
            
            plt.title(f"Histogram: {col}\n(Mean: {mean_val:.2f}, Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f})")
            plt.xlabel(col)
            plt.ylabel("Frequency (빈도)")
            plt.tight_layout()
            
            safe_col_name = col.replace(os.path.sep, '_').replace(':', '_')
            output_path = os.path.join(numeric_plot_dir, f"hist_{safe_col_name}.png")
            plt.savefig(output_path, dpi=90)
            plt.close()
            
            # 진행 상황 표시 (너무 많을 경우를 대비해 10개 단위로 출력)
            if i % 10 == 0:
                print(f"    - {i}/{total_num} 완료...")
                
        except Exception as e:
            print(f"    ❌ '{col}' 플롯 생성 실패: {e}")
            
    numeric_time = time.time() - start_numeric
    print(f"✓ 숫자형 컬럼 처리 완료 (소요 시간: {numeric_time:.2f}초)")

    # --- 범주형 컬럼 (카운트 플롯) 생성 ---
    print("\n... 2. 범주형 데이터 카운트 플롯 생성 중 ...")
    categorical_plot_dir = os.path.join(plot_dir, "categorical_countplots")
    os.makedirs(categorical_plot_dir, exist_ok=True)

    start_categorical = time.time()
    
    for i, col in enumerate(categorical_cols, 1):
        try:
            unique_count = df[col].nunique()
            # 너무 많은 고유값을 가진 범주형 데이터(예: 파일 경로 등)는 시각화에서 제외
            if unique_count > 50 or unique_count == 0:
                continue
        
            chart_width = max(10, unique_count * 0.5)
            plt.figure(figsize=(chart_width, 7))
            
            # 빈도수 순으로 정렬
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
    
    print("--- [시작] 원본 데이터 분포 확인 (필터링 없음) ---")
    
    # 1단계: 필터링 없이 데이터 로드 및 저장
    df_full = process_without_filter(
        csv_file=MERGED_CSV_FILE, 
        output_file=OUTPUT_CLEANED_CSV
    )
    
    # 2단계: 분포도 생성
    if df_full is not None:
        generate_plots(
            df=df_full, 
            plot_dir=CLEANED_PLOT_DIR
        )
    else:
        print("\n❌ 1단계에서 오류가 발생하여 2단계를 건너뜁니다.")

    print("\n" + "="*70)
    print("🎉 모든 작업 완료!")
    print(f"  - 저장된 CSV: {OUTPUT_CLEANED_CSV}")
    print(f"  - 분포도 폴더: {CLEANED_PLOT_DIR}")
    print("="*70)