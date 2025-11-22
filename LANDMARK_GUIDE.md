# 얼굴 랜드마크 추출 기능 가이드 (v5)

## 📋 개요
영상 업로드 시 백엔드에서 **실시간으로 얼굴 랜드마크를 추출한 영상을 생성**하는 기능이 추가되었습니다.

## 🎯 주요 기능
- **자동 랜드마크 추출**: 영상 업로드 시 자동으로 얼굴 랜드마크 추출
- **3초 처리 시간 제한**: 처리 시간을 약 3초로 제한하여 빠른 응답 (영상 길이는 원본 유지)
- **원본 속도 유지**: 배속 없이 원본 영상과 동일한 길이와 속도로 생성
- **MediaPipe 사용**: Google의 MediaPipe를 활용한 고정밀 얼굴 랜드마크 감지
- **다운로드 지원**: API를 통해 생성된 랜드마크 영상 다운로드 가능

## ⚠️  중요: 영상 길이에 대하여
- **원본 영상 길이 유지**: 랜드마크 영상은 원본과 동일한 길이로 생성됩니다
- **3초 = 처리 시간 제한**: 3초는 영상을 처리하는 최대 시간이며, 3초가 지나면 처리를 중단합니다
- **부분 처리 가능**: 긴 영상의 경우 앞부분만 처리되어 생성될 수 있습니다
- **예시**: 10초 영상을 3초 동안 처리하면, 약 앞 3~4초 정도만 랜드마크가 표시된 영상이 생성됩니다

## 🛠️ 설치 방법

### 1. 필요한 라이브러리 설치
```bash
# conda 환경 활성화 (이미 있다면)
conda activate deepfake_backend_env

# Python 라이브러리 설치
pip install opencv-python mediapipe numpy
```

또는 전체 requirements 재설치:
```bash
pip install -r requirements.txt
```

### 2. ffmpeg 설치 (필수!)
영상을 브라우저에서 재생 가능하도록 H.264로 재인코딩하려면 ffmpeg가 필요합니다.

**Windows:**
```bash
# Chocolatey 사용
choco install ffmpeg

# 또는 https://ffmpeg.org/download.html 에서 다운로드
# 환경변수 PATH에 추가
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**설치 확인:**
```bash
ffmpeg -version
```

### 3. 디렉토리 구조 확인
프로젝트에 다음 파일들이 추가/수정되었습니다:
```
app/
├── main.py                    # 수정됨 (static files 추가)
├── services/
│   └── landmark_extractor.py  # 새로 추가됨
├── routers/
│   └── detect.py              # 수정됨
├── models/
│   └── video.py               # 수정됨 (landmark_video_path 필드 추가)
└── schemas/
    └── video.py               # 수정됨 (DetectResult에 필드 추가)

deepfake_web/
└── views/
    └── detect.py              # 수정됨 (랜드마크 영상 표시 추가)
```

### 3. 주요 변경사항
- **app/main.py**: `/uploads` 경로로 static files serving 추가
- **원본 속도 유지**: 프레임 스킵 없이 시간 제한만 적용
- **영상 길이**: 원본과 동일한 길이 (처리 시간 내에서만 진행)

## 🚀 사용 방법

### 1. 서버 실행
```bash
# 백엔드 서버 시작
uvicorn app.main:app --reload

# 서버가 http://localhost:8000 에서 실행됩니다
```

### 2. API 사용

#### 📤 영상 업로드 및 랜드마크 추출
```bash
# curl을 이용한 테스트
curl -X POST "http://localhost:8000/detect/upload" \
  -F "file=@test_video.mp4" \
  -F "user_id=1"
```

**응답 예시:**
```json
{
  "video_id": 1,
  "is_deepfake": 0,
  "confidence": 0.85,
  "landmark_video_path": "uploads/landmarks/landmark_test_video.mp4",
  "landmark_info": {
    "processing_time": 2.45,
    "processed_frames": 90,
    "faces_detected": 87
  }
}
```

#### 📥 랜드마크 영상 다운로드
```bash
# 생성된 랜드마크 영상 다운로드
curl -O "http://localhost:8000/detect/landmark/1"
```

또는 브라우저에서:
```
http://localhost:8000/detect/landmark/1
```

### 3. Swagger UI에서 테스트
1. 브라우저에서 `http://localhost:8000/docs` 접속
2. `POST /detect/upload` 선택
3. "Try it out" 클릭
4. 파일 선택 후 "Execute" 클릭
5. 응답에서 `landmark_video_path` 확인
6. `GET /detect/landmark/{video_id}` 로 영상 다운로드

## 📊 처리 과정

### 랜드마크 추출 흐름
```
1. 영상 업로드
   ↓
2. 원본 영상 저장 (uploads/)
   ↓
3. 얼굴 랜드마크 추출 시작
   ├─ 프레임별로 처리
   ├─ MediaPipe로 얼굴 감지
   ├─ 468개 랜드마크 포인트 추출
   └─ 랜드마크 그리기
   ↓
4. 랜드마크 영상 저장 (uploads/landmarks/)
   ↓
5. DB에 경로 저장
   ↓
6. 응답 반환 (경로 + 처리 정보)
```

### 성능 최적화
- **시간 제한**: 3초 처리 시간을 초과하면 자동으로 중단
- **원본 속도 유지**: 프레임을 건너뛰지 않고 순차적으로 처리
- **단일 얼굴**: 한 번에 하나의 얼굴만 처리하여 속도 향상
- **효율적인 감지**: MediaPipe의 비디오 모드 사용

## 🔍 랜드마크 상세 정보

### MediaPipe Face Mesh
- **총 468개의 3D 랜드마크 포인트**
- 얼굴 윤곽, 눈, 코, 입, 눈썹 등 감지
- 실시간 처리 최적화

### 그려지는 요소
1. **FACEMESH_TESSELATION**: 얼굴 전체 메쉬
2. **FACEMESH_CONTOURS**: 얼굴 윤곽선
3. **FACEMESH_IRISES**: 눈동자

## 📁 파일 구조

### 입력 파일
- 위치: `uploads/`
- 이름: 원본 파일명 그대로

### 출력 파일
- 위치: `uploads/landmarks/`
- 이름: `landmark_{원본파일명}`
- 형식: MP4 (원본과 동일한 해상도)

## ⚙️ 설정 옵션

### landmark_extractor.py 커스터마이징
```python
# app/services/landmark_extractor.py

# 처리 시간 조절 (기본 3초)
create_landmark_video(
    input_path="video.mp4",
    max_processing_time=5.0  # 5초로 변경
)

# 최대 프레임 수 직접 지정
extractor = LandmarkExtractor()
result = extractor.extract_landmarks_from_video(
    input_path="video.mp4",
    output_path="output.mp4",
    max_frames=150  # 150프레임만 처리
)
```

### MediaPipe 설정 변경
```python
# app/services/landmark_extractor.py의 __init__ 메서드

self.face_mesh = self.mp_face_mesh.FaceMesh(
    max_num_faces=2,  # 여러 얼굴 감지
    min_detection_confidence=0.7,  # 감지 정확도 높이기
    min_tracking_confidence=0.7
)
```

## 🐛 문제 해결

### 1. 영상이 웹에서 재생되지 않음 (검은 화면) ⚠️
**원인**: 
- 비디오 코덱이 브라우저와 호환되지 않음
- ffmpeg가 설치되지 않아 H.264 재인코딩 실패

**해결**:
```bash
# 1. ffmpeg 설치 확인
ffmpeg -version

# 2. ffmpeg 설치 (없으면)
# Windows
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# 3. 서버 재시작
uvicorn app.main:app --reload
```

**확인 방법**:
- 서버 로그에서 "✅ ffmpeg 재인코딩 완료" 메시지 확인
- 다운로드한 파일을 로컬 플레이어로 재생 테스트
- 브라우저 개발자 도구 콘솔에서 에러 확인

### 2. OpenCV 설치 오류
```bash
# 헤드리스 버전 설치 (서버 환경)
pip install opencv-python-headless
```

### 2. MediaPipe 설치 오류
```bash
# 특정 버전 설치
pip install mediapipe==0.10.9
```

### 3. 영상이 생성되지 않음
- 입력 영상이 손상되었는지 확인
- 충분한 디스크 공간 확인
- 로그 확인: 서버 콘솔에 🎯, ✅, ⚠️  메시지 확인

### 4. 랜드마크가 감지되지 않음
- 영상에 얼굴이 명확하게 나오는지 확인
- 조명이 너무 어둡지 않은지 확인
- `min_detection_confidence` 값을 낮춰보기 (0.3~0.5)

### 5. 처리 시간이 너무 길 때
```python
# max_processing_time을 더 짧게 설정
create_landmark_video(
    input_path="video.mp4",
    max_processing_time=2.0  # 2초로 단축
)
```

## 📝 로그 확인

서버 실행 중 다음과 같은 로그가 출력됩니다:
```
🎯 랜드마크 추출 시작: uploads/test_video.mp4
✅ 랜드마크 영상 생성 완료: uploads/landmarks/landmark_test_video.mp4
   - 처리 시간: 2.45초
   - 처리 프레임: 90/300
```

## 🔗 API 엔드포인트

### POST /detect/upload
영상 업로드 및 분석 (랜드마크 자동 생성)

**요청:**
- file: 영상 파일 (multipart/form-data)
- user_id: 사용자 ID (optional)

**응답:**
- video_id: 비디오 ID
- is_deepfake: 딥페이크 여부 (0 or 1)
- confidence: 신뢰도 (0.0~1.0)
- landmark_video_path: 랜드마크 영상 경로
- landmark_info: 처리 정보 (시간, 프레임 수 등)

### POST /detect/youtube
YouTube 링크 분석 (랜드마크 자동 생성)

### GET /detect/landmark/{video_id}
랜드마크 영상 다운로드

## 🎓 추가 학습 자료

- [MediaPipe Face Mesh 문서](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## 💡 팁

1. **프론트엔드 통합**
   - 응답의 `landmark_video_path`를 사용해 영상 표시
   - `/detect/landmark/{video_id}` URL로 직접 재생 가능

2. **모델 팀을 위한 데이터**
   - 생성된 랜드마크 영상은 `uploads/landmarks/`에 저장
   - 원본과 함께 분석에 사용 가능
   - DB의 `landmark_video_path` 필드로 경로 추적

3. **성능 모니터링**
   - `landmark_info.processing_time`으로 처리 시간 확인
   - `landmark_info.faces_detected`로 얼굴 감지 성공률 확인

## 📞 문의

문제가 발생하거나 추가 기능이 필요하면 팀원에게 문의하세요!
