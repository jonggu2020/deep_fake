"""얼굴 랜드마크 추출 영상 생성 서비스.

MediaPipe를 사용하여 영상에서 얼굴 랜드마크를 추출하고,
랜드마크가 그려진 새로운 영상을 생성한다.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional
import time


class LandmarkExtractor:
    """얼굴 랜드마크 추출 및 영상 생성 클래스."""
    
    def __init__(self):
        """MediaPipe Face Mesh 초기화."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face Mesh 설정 (정적 이미지 모드 꺼서 비디오 처리 최적화)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # 한 얼굴만 처리 (속도 향상)
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks_from_video(
        self, 
        input_path: str, 
        output_path: str,
        max_processing_time: Optional[float] = None,
        target_fps: Optional[int] = None
    ) -> dict:
        """영상에서 얼굴 랜드마크를 추출하여 새로운 영상으로 저장.
        
        Args:
            input_path: 입력 영상 파일 경로
            output_path: 출력 영상 파일 경로
            max_processing_time: 최대 처리 시간 (초) - 이 시간 초과 시 중단
            target_fps: 출력 영상의 FPS (None이면 원본과 동일)
        
        Returns:
            dict: 처리 결과 정보
                - success: 성공 여부
                - processed_frames: 처리된 프레임 수
                - total_frames: 전체 프레임 수
                - processing_time: 처리 시간 (초)
                - output_path: 출력 파일 경로
                - faces_detected: 얼굴이 감지된 프레임 수
        """
        start_time = time.time()
        
        # 입력 영상 열기
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {
                "success": False,
                "error": "Failed to open video file"
            }
        
        # 영상 속성 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 FPS 설정 (원본과 동일하게)
        output_fps = target_fps if target_fps else fps
        
        # VideoWriter 설정 - H.264 코덱 사용 (브라우저 호환)
        # x264 코덱을 사용하면 웹 브라우저에서 재생 가능
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # avc1이 안 되면 mp4v로 폴백
        if not out.isOpened():
            print("⚠️  avc1 코덱 실패, mp4v로 재시도...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return {
                "success": False,
                "error": "Failed to create output video file"
            }
        
        processed_frames = 0
        faces_detected = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 시간 제한 체크 (지정된 시간 초과 시 중단)
                if max_processing_time and (time.time() - start_time) > max_processing_time:
                    print(f"⏱️  최대 처리 시간({max_processing_time}초) 도달. 처리 중단.")
                    break
                
                # RGB로 변환 (MediaPipe는 RGB 사용)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 얼굴 랜드마크 추출
                results = self.face_mesh.process(rgb_frame)
                
                # 랜드마크가 감지되면 그리기
                if results.multi_face_landmarks:
                    faces_detected += 1
                    for face_landmarks in results.multi_face_landmarks:
                        # 랜드마크 그리기
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        
                        # 윤곽선 강조
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        
                        # 눈동자 강조
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )
                
                # 프레임 저장
                out.write(frame)
                processed_frames += 1
                
        finally:
            cap.release()
            out.release()
        
        processing_time = time.time() - start_time
        
        # 브라우저 호환성을 위해 ffmpeg로 재인코딩
        try:
            import subprocess
            import platform
            from app.core.config import settings
            
            temp_output = output_path + ".temp.mp4"
            
            # 원본을 temp로 이동
            Path(output_path).rename(temp_output)
            
            # ffmpeg 실행 파일 경로 찾기
            ffmpeg_cmd = 'ffmpeg'
            
            # 설정 파일에 경로가 지정되어 있으면 사용
            if settings.FFMPEG_PATH:
                ffmpeg_cmd = settings.FFMPEG_PATH
                print(f"📁 설정된 ffmpeg 경로 사용: {ffmpeg_cmd}")
            # Windows에서 ffmpeg가 PATH에 없을 경우 직접 경로 지정
            elif platform.system() == 'Windows':
                # 일반적인 설치 경로들 체크
                possible_paths = [
                    'ffmpeg',  # PATH에 있는 경우
                    r'C:\ffmpeg\bin\ffmpeg.exe',
                    r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                    r'C:\대학교 폴더\프로젝트 응용\딥페이크\ffmpeg-8.0-full_build\bin\ffmpeg.exe',
                ]
                
                for path in possible_paths:
                    try:
                        result = subprocess.run([path, '-version'], 
                                              capture_output=True, 
                                              timeout=5)
                        if result.returncode == 0:
                            ffmpeg_cmd = path
                            print(f"✅ ffmpeg 발견: {path}")
                            break
                    except:
                        continue
            
            # ffmpeg로 H.264 재인코딩
            ffmpeg_full_cmd = [
                ffmpeg_cmd,
                '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_full_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # 성공 - temp 파일 삭제
                Path(temp_output).unlink()
                print("✅ ffmpeg 재인코딩 완료 (브라우저 호환)")
            else:
                # 실패 - temp를 원본으로 복원
                print(f"⚠️  ffmpeg 재인코딩 실패, 원본 사용: {result.stderr}")
                Path(temp_output).rename(output_path)
        except Exception as e:
            print(f"⚠️  ffmpeg 후처리 실패 (ffmpeg 미설치?): {e}")
            # temp 파일이 있으면 원본으로 복원
            temp_path = Path(output_path + ".temp.mp4")
            if temp_path.exists():
                temp_path.rename(output_path)
        
        return {
            "success": True,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "processing_time": round(processing_time, 2),
            "output_path": output_path,
            "faces_detected": faces_detected,
            "fps": output_fps,
            "resolution": f"{width}x{height}"
        }
    
    def __del__(self):
        """리소스 정리."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def create_landmark_video(
    input_path: str,
    output_dir: str = "uploads/landmarks",
    max_processing_time: float = 3.0
) -> dict:
    """영상에서 랜드마크 추출 영상을 생성하는 편의 함수.
    
    Args:
        input_path: 입력 영상 경로
        output_dir: 출력 디렉토리
        max_processing_time: 최대 처리 시간 (초)
    
    Returns:
        dict: 처리 결과
    """
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 출력 파일명 생성
    input_file = Path(input_path)
    output_filename = f"landmark_{input_file.stem}{input_file.suffix}"
    output_file = output_path / output_filename
    
    # 랜드마크 추출 (시간 제한만 적용)
    extractor = LandmarkExtractor()
    result = extractor.extract_landmarks_from_video(
        input_path=input_path,
        output_path=str(output_file),
        max_processing_time=max_processing_time
    )
    
    if result["success"]:
        result["output_filename"] = output_filename
    
    return result


# 간단한 테스트용 메인
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python landmark_extractor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = create_landmark_video(video_path)
    
    if result["success"]:
        print(f"✅ 랜드마크 영상 생성 완료!")
        print(f"   - 출력 파일: {result['output_path']}")
        print(f"   - 처리 프레임: {result['processed_frames']}/{result['total_frames']}")
        print(f"   - 얼굴 감지: {result['faces_detected']}프레임")
        print(f"   - 처리 시간: {result['processing_time']}초")
    else:
        print(f"❌ 실패: {result.get('error', 'Unknown error')}")
