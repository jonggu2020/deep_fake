"""얼굴 랜드마크 추출 및 시각화 기능.

MediaPipe를 사용하여 영상에서 얼굴 랜드마크를 추출하고,
랜드마크가 그려진 새로운 영상을 생성합니다.
"""

import cv2
import mediapipe as mp
from pathlib import Path
import subprocess
import os
import numpy as np


def create_landmark_video(input_video_path: str, max_duration: int = 3) -> str:
    """영상에서 얼굴 랜드마크를 추출하여 시각화된 영상을 생성합니다.
    
    Args:
        input_video_path: 입력 영상 경로
        max_duration: 최대 처리 시간 (초), 기본값 3초
        
    Returns:
        생성된 랜드마크 영상의 경로
    """
    # MediaPipe FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_detection = mp.solutions.face_detection
    
    # 입력 영상 열기
    cap = cv2.VideoCapture(input_video_path)
    
    # 영상 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 디코딩 실패 시 기본값 적용 (일부 mp4 코덱/FPS 인식 실패 방지)
    if fps <= 0:
        fps = 25
    if width <= 0 or height <= 0:
        # 기본 640x360 캔버스
        width, height = 640, 360
    
    # 출력 파일 경로 생성 (임시 파일 먼저 생성)
    input_path = Path(input_video_path)
    temp_path = input_path.parent / f"{input_path.stem}_landmark_temp.mp4"
    output_path = input_path.parent / f"{input_path.stem}_landmark{input_path.suffix}"
    
    # 비디오 라이터 설정 (임시로 mp4v 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    
    # 최대 프레임 수 계산 (max_duration초)
    max_frames = fps * max_duration
    frame_count = 0
    written_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detector:
        
        while cap.isOpened() and frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                # 디코딩 초기 실패 시 한 번 더 시도
                if frame_count == 0:
                    cap.release()
                    cap = cv2.VideoCapture(input_video_path)
                    success, frame = cap.read()
                if not success:
                    break
            # 프레임 사이즈 이상치 교정
            if frame is not None and (frame.shape[1] != width or frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height))
            
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 랜드마크 탐지
            results = face_mesh.process(rgb_frame)
            
            # 랜드마크 그리기
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
                # 랜드마크가 그려진 프레임
                pass
            else:
                # FaceMesh 미검출 시 FaceDetection 시도
                detection_results = face_detector.process(rgb_frame)
                if detection_results.detections:
                    for det in detection_results.detections:
                        location = det.location_data
                        if location and location.relative_bounding_box:
                            rb = location.relative_bounding_box
                            x1 = int(rb.xmin * width)
                            y1 = int(rb.ymin * height)
                            x2 = int((rb.xmin + rb.width) * width)
                            y2 = int((rb.ymin + rb.height) * height)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, 'FACE DETECTED - NO LANDMARK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    # 얼굴 박스만 표시된 프레임
                    pass
                else:
                    # 완전 미검출 프레임 안내 텍스트 (너무 반복되지 않게 첫 1초만 표시)
                    if frame_count < fps:
                        cv2.putText(frame, 'NO FACE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            # 프레임 저장
            out.write(frame)
            frame_count += 1
            written_frames += 1
    
    # 리소스 해제
    cap.release()
    out.release()
    
    # 프레임 하나도 기록 못했으면 placeholder 생성
    if written_frames == 0:
        # 최소 1초 길이(5fps, 5프레임) placeholder 생성 -> 플레이어 0:00 문제 감소
        placeholder_writer = cv2.VideoWriter(str(temp_path), fourcc, 5, (width, height))
        for i in range(5):
            placeholder_frame = np.zeros((height, width, 3), dtype=np.uint8)
            placeholder_frame[:] = (25, 25, 25)
            cv2.putText(placeholder_frame, 'NO FRAMES DECODED', (30, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            placeholder_writer.write(placeholder_frame)
        placeholder_writer.release()
    
    # ffmpeg로 H.264 변환 (브라우저 재생 호환성)
    try:
        # ffmpeg 명령어로 H.264 재인코딩 (faststart로 웹 재생 향상)
        proc = subprocess.run([
            'ffmpeg', '-i', str(temp_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', str(output_path)
        ], check=True, capture_output=True, timeout=45)
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        # ffmpeg 실패 시 stderr 출력 후 temp 파일 사용
        try:
            if hasattr(e, 'stderr') and e.stderr:
                print('[landmark ffmpeg error]', e.stderr.decode(errors='ignore'))
        except Exception:
            pass
        if os.path.exists(temp_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
    
    # 전체 처리 구간에서 아무 표시도 없었다면 첫 프레임에 안내 텍스트 삽입을 위해 재처리 (간단화: 이미 처리된 영상 그대로 사용)
    return str(output_path).replace('\\', '/')
