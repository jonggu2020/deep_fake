import os
import pandas as pd
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

def set_korean_font():
    """
    Matplotlib에서 한글을 지원하기 위한 폰트 설정 (Windows, macOS, Linux 대응)
    """
    # 사용 가능한 폰트 리스트 확인 (디버깅용, 평소엔 주석 처리)
    # print([f.name for f in fm.fontManager.ttflist])DWQQㅁ123ㅁ11
    
    font_name = None
    if os.name == 'nt':  # Windows
        font_name = 'Malgun Gothic'
    elif os.name == 'posix':
        if 'darwin' in os.uname().sysname.lower():  # macOS
            font_name = 'AppleGothic'
        else:  # Linux
            # "NanumGothic" 또는 "NanumBarunGothic" 등 설치된 폰트
            # apt-get install fonts-nanum* 등으로 설치 필요
            font_name = 'NanumGothic' 
            
    if font_name and font_name in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
        print(f"한글 폰트 '{font_name}'로 설정되었습니다.")
    else:
        print("경고: 'Malgun Gothic' (Windows) 또는 'AppleGothic' (macOS) 또는 'NanumGothic' (Linux)을 찾을 수 없습니다. 차트의 한글이 깨질 수 있습니다.")
        if os.name == 'nt':
             print("Windows 사용자의 경우 '맑은 고딕' 폰트가 설치되어 있는지 확인하세요.")
        elif os.name == 'posix' and 'darwin' not in os.uname().sysname.lower():
             print("Linux 사용자의 경우 'sudo apt-get install fonts-nanum*' 등으로 나눔폰트를 설치하세요.")


def analyze_video_properties(folder_path):
    """
    폴더 내의 비디오 파일들의 해상도, FPS, 시간을 분석합니다.
    """
    video_data = []
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
    
    print(f"'{folder_path}' 폴더 분석 시작...")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 'with' 구문으로 파일을 열어 리소스 자동 해제
                with VideoFileClip(file_path) as clip:
                    # 속성 추출
                    width, height = clip.size
                    fps = clip.fps
                    duration_sec = clip.duration
                    
                    # 데이터 리스트에 추가
                    video_data.append({
                        'filename': filename,
                        'width': width,
                        'height': height,
                        'resolution': f"{width}x{height}", # 해상도 문자열
                        'fps': round(fps, 2), # FPS (소수점 2자리까지)
                        'duration_sec': duration_sec,
                        'duration_min': round(duration_sec / 60, 2) # 분 단위
                    })
                print(f"[성공] {filename} ( {width}x{height}, {fps:.2f}fps, {duration_sec:.1f}초 )")
                
            except Exception as e:
                print(f"[!!오류!!] '{filename}' 파일 분석 중 오류 발생: {e}")
                
    if not video_data:
        print("분석할 비디오 파일이 없습니다.")
        return None

    # 리스트를 Pandas DataFrame으로 변환
    df = pd.DataFrame(video_data)
    return df

def plot_video_distributions(df):
    """
    분석된 비디오 속성 데이터프레임을 받아 분포도를 시각화하고 파일로 저장합니다.
    """
    if df is None or df.empty:
        print("플로팅할 데이터가 없습니다.")
        return

    print("\n--- 차트 생성 중 ---")
    
    # Seaborn 스타일 설정
    sns.set_style("whitegrid")
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('비디오 속성 분포도', fontsize=20, y=1.03)

    # 1. 해상도 분포 (가로 막대 차트)
    sns.countplot(
        y='resolution', 
        data=df, 
        order=df['resolution'].value_counts().index, 
        ax=axes[0, 0],
        palette='viridis'
    )
    axes[0, 0].set_title('해상도 (Resolution) 분포', fontsize=14)
    axes[0, 0].set_xlabel('파일 개수', fontsize=12)
    axes[0, 0].set_ylabel('해상도', fontsize=12)

    # 2. FPS 분포 (세로 막대 차트)
    sns.countplot(
        x='fps', 
        data=df, 
        order=df['fps'].value_counts().index, 
        ax=axes[0, 1],
        palette='plasma'
    )
    axes[0, 1].set_title('FPS (초당 프레임) 분포', fontsize=14)
    axes[0, 1].set_xlabel('FPS', fontsize=12)
    axes[0, 1].set_ylabel('파일 개수', fontsize=12)
    # x축 라벨이 많을 경우 겹칠 수 있으므로 45도 회전
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. 재생 시간 분포 (히스토그램, 분 단위)
    sns.histplot(
        data=df, 
        x='duration_min', 
        kde=True,  # 밀도 곡선 표시
        ax=axes[1, 0],
        color='steelblue'
    )
    axes[1, 0].set_title('재생 시간 분포 (분 단위)', fontsize=14)
    axes[1, 0].set_xlabel('재생 시간 (분)', fontsize=12)
    axes[1, 0].set_ylabel('파일 개수', fontsize=12)

    # 4. 세로 해상도 (Height) 분포 (막대 차트)
    # (1080, 720, 420 등 변환 목표와 직접 비교하기 좋음)
    sns.countplot(
        x='height', 
        data=df, 
        order=df['height'].value_counts().index.sort_values(ascending=False), # 해상도 크기 순 정렬
        ax=axes[1, 1],
        palette='cividis'
    )
    axes[1, 1].set_title('세로 해상도 (Height) 분포', fontsize=14)
    axes[1, 1].set_xlabel('세로 해상도 (p)', fontsize=12)
    axes[1, 1].set_ylabel('파일 개수', fontsize=12)

    # 레이아웃 정리
    plt.tight_layout()
    
    # 파일로 저장
    save_path = 'video_analysis_distribution.png'
    plt.savefig(save_path, dpi=100)
    
    print(f"차트가 '{save_path}' 파일로 저장되었습니다.")


# --- 메인 코드 실행 ---t
if __name__ == "__main__":
    
    # 0. 한글 폰트 설정
    set_korean_font()
    
    # 1. 원본 영상이 있는 폴더 경로 (수정 필요!!)
    SOURCE_FOLDER = "../data" 
    
    # 2. 비디오 속성 분석
    video_df = analyze_video_properties(SOURCE_FOLDER)
    
    if video_df is not None:
        # 3. 분석 결과 요약 출력
        print("\n--- 데이터 분석 요약 ---")
        print(video_df.info())
        
        print("\n[기술 통계 (수치형 데이터)]")
        # .describe()는 수치형 데이터(fps, 시간, 크기)의 통계를 보여줍니다.
        print(video_df.describe())
        
        print("\n[해상도별 개수 (상위 10개)]")
        print(video_df['resolution'].value_counts().head(10))
        
        print("\n[FPS별 개수]")
        print(video_df['fps'].value_counts())
        
        # 4. 시각화 함수 호출
        plot_video_distributions(video_df)
        
    print("\n모든 작업이 완료되었습니다.")
