import os
import subprocess

def convert_videos_ffmpeg(input_folder, output_folder, target_height=720):
    """
    FFmpeg를 직접 호출하여 비디오를 저해상도로 변환합니다. (고속)

    :param input_folder: 원본 비디오가 있는 폴더
    :param output_folder: 변환된 비디오를 저장할 폴더
    :param target_height: 목표 해상도 (세로 픽셀, 예: 720 또는 420)
    """
    
    # 지원할 비디오 확장자 목록
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    # 입력 폴더 내의 모든 파일 순회
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(video_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"'{filename}' 변환 중 (FFmpeg)...")

            try:
                # FFmpeg 명령어 구성
                # -i : 입력 파일
                # -vf "scale=-2:720" : 비디오 필터. 세로 720px로 맞추고(-2는 가로 비율 자동 계산)
                # -c:v libx264 : 비디오 코덱 H.264
                # -c:a aac : 오디오 코덱 AAC (원본 오디오 복사시 -c:a copy)
                # -preset medium : 인코딩 속도/품질 균형 (ultrafast, fast, medium, slow)
                # -crf 23 : 품질 설정 (18~28 사이 권장. 낮을수록 고화질/고용량)
                # -y : 덮어쓰기 허용
                command = [
                    'ffmpeg',
                    '-i', input_path,
                    '-vf', f'scale=-2:{target_height}',  # 비율 유지하며 세로 해상도 맞춤
                    '-c:v', 'libx264',
                    '-preset', 'fast',  # 속도 우선
                    '-crf', '24',        # 적절한 품질 (숫자가 클수록 압축률이 높음)
                    '-c:a', 'aac',        # 오디오 코덱
                    '-b:a', '128k',       # 오디오 비트레이트
                    '-y',                 # 대상 파일이 있어도 덮어쓰기
                    output_path
                ]
                
                # FFmpeg 명령어 실행
                # capture_output=True, text=True로 설정하면 상세 출력을 볼 수 있음
                # 여기서는 오류만 표기하도록 hide_banner와 loglevel 설정 추가
                command.insert(1, '-hide_banner')
                command.insert(2, '-loglevel')
                command.insert(3, 'error') # 오류만 표시 (진행상황 보려면 'info')

                subprocess.run(command, check=True, encoding='utf-8')
                
                print(f"'{filename}' -> {target_height}p 변환 완료 (FFmpeg).")

            except subprocess.CalledProcessError as e:
                print(f"'{filename}' 변환 중 FFmpeg 오류 발생: {e}")
            except Exception as e:
                print(f"'{filename}' 처리 중 알 수 없는 오류: {e}")

# --- 실행 ---
if __name__ == "__main__":
    # 1. 원본 영상이 있는 폴더 경로 (수정 필요)
    SOURCE_FOLDER = "../data" 
    
    # 2. 변환된 영상을 저장할 폴더 경로 (수정 필요)
    DEST_FOLDER = "../test"
    
    # 3. 목표 해상도 (720p 또는 480p)
    TARGET_RESOLUTION_P = 240 # 240 변경 테스트
    
    print(f"비디오 변환 시작 (FFmpeg): {SOURCE_FOLDER} -> {DEST_FOLDER} ({TARGET_RESOLUTION_P}p)")
    
    convert_videos_ffmpeg(SOURCE_FOLDER, DEST_FOLDER, TARGET_RESOLUTION_P)
    
    print("모든 작업이 완료되었습니다.")