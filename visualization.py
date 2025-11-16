import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

# Matplotlib에서 수많은 플롯을 생성할 때 발생하는 경고를 억제합니다.
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.pyplot')

# --- 1. 사용자 설정 ---

# ⚠️ [수정필요]
# 이전 단계에서 통합한 'master_summary_v1.csv' 또는 'master_summary_v2_optimized.csv'
# 파일의 정확한 경로를 지정하세요.
MERGED_CSV_FILE = "./master_summary_v1_standard.csv" 

# ⚠️ 모든 그래프가 저장될 폴더 이름
OUTPUT_PLOT_DIR = "./distribution_plots"

# ---

def analyze_distributions(csv_file, plot_dir):
    """
    통합된 CSV 파일의 모든 컬럼을 읽어 분포도를 시각화합니다.
    - 숫자형 컬럼: 히스토그램 (Histogram)
    - 범주형 컬럼: 카운트 플롯 (Count Plot)
    """
    print("="*70)
    print("📊 데이터 분포도 분석 스크립트 시작")
    print("="*70)
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- 2. 데이터 로드 ---
    print(f"'{csv_file}' 파일을 불러오는 중...")
    print("⚠️ (주의) 파일이 매우 크면(예: 수십 GB) 메모리 부족으로 멈출 수 있습니다.")
    
    start_load = time.time()
    try:
        df = pd.read_csv(csv_file)
        load_time = time.time() - start_load
        print(f"✓ 데이터 로드 완료 (총 {len(df)} 행, {len(df.columns)} 열) / 소요시간: {load_time:.2f}초")
    
    except MemoryError:
        print("❌ [메모리 오류!] 파일이 너무 커서 RAM에 모두 올릴 수 없습니다.")
        print("   이 스크립트의 32행 근처 'df = pd.read_csv(csv_file)' 코드를")
        print("   다음과 같이 수정하여 데이터의 일부만 '샘플링'하여 분석해보세요:")
        print("\n   (예시) 10만 개 행만 무작위로 샘플링하여 분석하기")
        print("   df = pd.read_csv(csv_file).sample(n=100000, random_state=42)")
        print("\n   (예시) 전체 데이터의 10%만 무작위로 샘플링하여 분석하기")
        print("   df = pd.read_csv(csv_file).sample(frac=0.1, random_state=42)")
        return
        
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        print("   'MERGED_CSV_FILE' 경로가 올바른지 확인하세요.")
        return
        
    # --- 3. 컬럼 유형 분리 ---
    # 숫자형(float64, int64 등) 컬럼 선택
    numerical_cols = df.select_dtypes(include=['number']).columns
    # 문자열/범주형(object, category) 컬럼 선택
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    print(f"\n--- 컬럼 유형 분석 ---")
    print(f"  - 숫자형 컬럼 (히스토그램 생성 대상): {len(numerical_cols)}개")
    print(f"  - 범주형 컬럼 (카운트 플롯 생성 대상): {len(categorical_cols)}개")

    # --- 4. 숫자형 컬럼 (히스토그램) 생성 ---
    print("\n... 1. 숫자형 데이터 히스토그램 생성 중 ...")
    numeric_plot_dir = os.path.join(plot_dir, "numerical_histograms")
    os.makedirs(numeric_plot_dir, exist_ok=True)
    
    total_num = len(numerical_cols)
    start_numeric = time.time()
    
    for i, col in enumerate(numerical_cols, 1):
        print(f"  [{i}/{total_num}] '{col}' 처리 중...")
        try:
            # NaN (결측치) 값을 제외하고 데이터를 가져옴
            data_series = df[col].dropna()
            
            if data_series.empty:
                print(f"    -> (경고) {col} 컬럼은 데이터가 모두 NaN이므로 건너뜁니다.")
                continue
            
            plt.figure(figsize=(10, 6))
            # Seaborn의 histplot을 사용하면 KDE(밀도 곡선)를 함께 그릴 수 있음
            sns.histplot(data_series, kde=True, bins=50) 
            
            mean_val = data_series.mean()
            std_val = data_series.std()
            
            plt.title(f"Histogram: {col}\n(Mean: {mean_val:.2f}, Std: {std_val:.2f})")
            plt.xlabel(col)
            plt.ylabel("Frequency (빈도)")
            plt.tight_layout()
            
            # 파일 이름에 특수문자가 포함될 수 있으므로 안전하게 처리 (예: /)
            safe_col_name = col.replace(os.path.sep, '_').replace(':', '_')
            output_path = os.path.join(numeric_plot_dir, f"hist_{safe_col_name}.png")
            
            plt.savefig(output_path, dpi=90) # 해상도(dpi)를 낮춰 파일 크기↓, 속도↑
            plt.close() # 메모리 해제를 위해 플롯을 닫음
            
        except Exception as e:
            print(f"    ❌ '{col}' 플롯 생성 실패: {e}")
            
    numeric_time = time.time() - start_numeric
    print(f"✓ 숫자형 컬럼 처리 완료 (소요 시간: {numeric_time:.2f}초)")

    # --- 5. 범주형 컬럼 (카운트 플롯) 생성 ---
    print("\n... 2. 범주형 데이터 카운트 플롯 생성 중 ...")
    categorical_plot_dir = os.path.join(plot_dir, "categorical_countplots")
    os.makedirs(categorical_plot_dir, exist_ok=True)

    total_cat = len(categorical_cols)
    start_categorical = time.time()
    
    for i, col in enumerate(categorical_cols, 1):
        print(f"  [{i}/{total_cat}] '{col}' 처리 중...")
        
        try:
            # 고유값(Unique value) 개수 확인
            unique_count = df[col].nunique()
            
            # 'video_id' 처럼 고유값이 너무 많으면(예: 100개 초과) 플롯이 불가능
            if unique_count > 100:
                print(f"    -> (생략) 고유값이 {unique_count}개로 너무 많습니다. ('video_id' 등)")
                continue
            
            if unique_count == 0 or df[col].isnull().all():
                print(f"    -> (생략) 데이터가 없습니다.")
                continue
        
            # 고유값 개수에 따라 차트의 폭을 동적으로 조절
            chart_width = max(10, unique_count * 0.5)
            plt.figure(figsize=(chart_width, 7))
            
            # 값(value)의 빈도순(DESC)으로 정렬하여 플롯팅
            order = df[col].value_counts().index
            sns.countplot(data=df, x=col, order=order)
            
            plt.title(f"Count Plot: {col} (Unique Values: {unique_count})")
            plt.xlabel(col)
            plt.ylabel("Count (개수)")
            
            # x축 레이블이 길거나 많으면 겹치므로 45도 회전
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

    print("\n" + "="*70)
    print("🎉 모든 분포도 분석이 완료되었습니다.")
    print(f"결과물은 '{plot_dir}' 폴더 내의 하위 폴더에서 확인하세요.")
    print(f"  - 숫자형 플롯: {numeric_plot_dir}")
    print(f"  - 범주형 플롯: {categorical_plot_dir}")
    print("="*70)

if __name__ == "__main__":
    # 스크립트 실행
    analyze_distributions(csv_file=MERGED_CSV_FILE, plot_dir=OUTPUT_PLOT_DIR)