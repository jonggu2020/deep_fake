# Deepfake Detection Backend

시공간 + 음성 정보를 이용한 딥페이크 탐지 프로젝트의 **백엔드 서버**입니다.  
FastAPI 기반으로 만들어졌고, 서버/네트워크를 잘 모르는 사람도 바로 실행하고 테스트할 수 있도록 구성되어 있습니다.

## 1. 이 백엔드가 하는 일
- 영상 또는 유튜브 링크를 받아 **딥페이크 여부를 분석**하는 API 뼈대 제공  
- 현재는 모델이 없기 때문에 `inference.py`에서 **랜덤 결과**를 반환  
- 나중에 모델 팀(XGBoost, LSTM, CNN, AV-sync)이 모델을 완성하면  
  → `inference.py`를 교체하여 실제 서비스로 확장 가능

## 2. 기술 스택
- Python 3.10
- FastAPI
- Uvicorn
- SQLite
- SQLAlchemy
- Pydantic / pydantic-settings
- Passlib[bcrypt]
- yt-dlp (YouTube 다운로드)
- ngrok
- mediapipe (FaceMesh/FaceDetection)
- opencv-python (프레임 처리)
- numpy (mediapipe 호환: 1.24.x 권장)

## 3. 프로젝트 구조
```
deepfake_backend/
├─ app/
│  ├─ main.py
│  ├─ core/config.py
│  ├─ database.py
│  ├─ models/
│  │  ├─ user.py
│  │  └─ video.py
│  ├─ schemas/
│  │  ├─ user.py
│  │  └─ video.py
│  ├─ routers/
│  │  ├─ auth.py
│  │  └─ detect.py
│  └─ services/
│     ├─ inference.py
│     ├─ youtube.py
│     └─ firebase_logger.py
├─ deepfake_web/          # 프론트엔드 (Streamlit)
│  ├─ main.py
│  ├─ views/
│  └─ services/
├─ uploads/
├─ secrets/                # Firebase 키 저장 (Git 제외)
├─ .env.example
├─ INTEGRATION_GUIDE.md
└─ requirements.txt
```

## 4. 파일 설명
### ▶ app/main.py
FastAPI 서버 시작점 (DB 생성, CORS, 라우터 등록)

### ▶ app/core/config.py
프로젝트 설정 관리

### ▶ app/database.py
SQLite DB 연결, 세션관리, Base 생성

### ▶ app/models/
SQLAlchemy ORM 모델

### ▶ app/schemas/
Pydantic 요청/응답 모델

### ▶ app/routers/
/auth, /detect API 정의

### ▶ app/services/
- **inference.py**: 분석 로직(현재 랜덤, 향후 실제 모델로 교체)
- **youtube.py**: 유튜브 다운로드
- **firebase_logger.py**: Firebase RTDB에 탐지 로그 저장

### ▶ deepfake_web/
Streamlit 기반 웹 프론트엔드
- 회원가입/로그인 UI
- 영상 업로드 및 YouTube 링크 입력 UI
- 탐지 결과 시각화

## 5. 개발 환경 세팅
### 1) 가상환경 (선택)
```bash
conda create -n deepfake_backend_env python=3.10
conda activate deepfake_backend_env
```

### 2) 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3) 환경 변수 설정 (선택)
Firebase/MySQL 사용 시 `.env.example`을 복사하여 `.env` 파일 생성:
```bash
cp .env.example .env
# 편집기로 .env 열어서 필요한 값 입력
```

Firebase 사용 시 서비스 계정 키 배치:
```
secrets/firebase-service-account.json
```

## 6. 서버 실행
### 로컬 테스트 (기본)
```bash
# 터미널 1: 백엔드 API 서버
uvicorn app.main:app --reload

# 터미널 2: 프론트엔드 (Streamlit)
cd deepfake_web
streamlit run main.py
```

접속:
- 백엔드 API: http://localhost:8000
- 백엔드 문서: http://localhost:8000/docs
- 프론트엔드: http://localhost:8501

### 외부 접속 (ngrok 사용)
```bash
# 터미널 1: 백엔드
uvicorn app.main:app --reload --port 8000

# 터미널 2: ngrok으로 백엔드 터널링
# ngrok 파일 위치로 cd 이동 후 .\ngrok.exe http 8000
.\ngrok http 8000
# 출력된 URL 복사 (예: https://xxxx-xxxx.ngrok-free.app)

# 터미널 3: 프론트엔드
cd deepfake_web
streamlit run main.py --server.port 8501
```

**프론트엔드 사용법:**
1. 브라우저에서 http://localhost:8501 접속
2. 왼쪽 사이드바 "Backend Base URL"에 ngrok URL 입력
3. Auth 메뉴에서 회원가입/로그인
4. Detect 메뉴에서 영상 업로드 또는 YouTube 링크 입력

**외부 사용자 공유:**
- ngrok URL을 공유하면 외부에서 백엔드 API 직접 호출 가능
- 프론트엔드는 로컬에서만 실행 (또는 별도 배포 필요)

## 7. 제공 API
### 인증
- `POST /auth/signup` - 회원가입
- `POST /auth/login` - 로그인

### 탐지
- `POST /detect/upload` - 파일 업로드 탐지
- `POST /detect/youtube` - YouTube 링크 탐지

### 기타
- `GET /` - 헬스 체크

**Swagger 문서:**
- 로컬: http://localhost:8000/docs
- ngrok: https://xxxx-xxxx.ngrok-free.app/docs

## 8. 향후 확장
- 실제 모델 연결 (XGBoost, LSTM, CNN)
- MySQL/Firebase 연동 (현재 선택적 지원)
- JWT 인증 강화
- 로깅/예외 처리 추가
- 프론트엔드 배포 (Streamlit Cloud 등)

## 9. 통합 가이드 요약
프론트(HOTTI) + Firebase 로그 저장 + 선택적 MySQL 사용을 위한 상세 절차는 `INTEGRATION_GUIDE.md` 참고.

핵심 요약:
- `.env` 파일 작성 (MySQL/Firebase 경로 설정)
- `secrets/firebase-service-account.json` 키 파일 배치 (Git 추적 제외)
- 필요 시 `MYSQL_URL` 환경변수로 MySQL 활성화 (미설정 시 SQLite)
- 업로드/유튜브 탐지 후 Firebase 로그 자동 기록 (키 없으면 건너뜀)
- `DeepFake_DB/DB_test.py`로 연동 테스트 가능

## 10. 데이터베이스 및 Firebase 연동

### 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일 생성:
```env
# Firebase 설정
FIREBASE_CREDENTIALS=secrets/firebase-service-account.json
FIREBASE_DATABASE_URL=https://sw-deepfake-project-default-rtdb.firebaseio.com/
ENABLE_FIREBASE_LOG=1

# MySQL 설정 (준규 DB 서버 연동)
MYSQL_URL=mysql+pymysql://root:PASSWORD@172.30.1.60:3306/firebase_db_tset

# 로컬 테스트용 SQLite (MySQL 연결 안 될 때)
# MYSQL_URL=sqlite:///./test_firebase.db
```

### MySQL 연동 정보
- **Host:** 172.30.1.60 (준규 DB 서버 외부 IP)
- **Port:** 3306
- **User:** root
- **Database:** firebase_db_tset
- **주의:** 외부 접속 허용됨, 로컬 테스트 시 localhost 대신 172.30.1.60 사용 가능

### Firebase 서비스 계정 키
1. Firebase Console에서 서비스 계정 키 다운로드
2. `secrets/firebase-service-account.json` 경로에 저장
3. `.gitignore`에 `secrets/` 포함되어 Git 추적 제외됨

### 테스트 실행
```bash
# DB + Firebase 연동 테스트
python DeepFake_DB/DB_test.py
```

성공 시 Firebase Realtime Database의 `/detection_logs`에 데이터가 저장됩니다.

## 11. 최근 업데이트

### 2025-11-22: 얼굴 랜드마크 추출 기능(v5) 안정화 🎯
- **구현 파일:** `app/services/landmark.py` (FaceMesh + FaceDetection fallback, ffmpeg 재인코딩)
- **응답 필드:** `landmark_video_path` (정적 `/uploads` 경로)
- **처리 범위:** 앞부분 최대 3초 프레임만 빠르게 분석 후 그린 결과 영상 생성
- **재생 안정화:** ffmpeg H.264 (`libx264`, `+faststart`) 변환 및 실패 시 원본 mp4v 사용
- **Fallback:** 얼굴 미검출 시 'NO FACE' 또는 박스 표시, 디코딩 실패 시 placeholder 영상 생성
- **사용 가이드:** `LANDMARK_GUIDE.md` 참고 (세부 설정 및 문제 해결)

### 2025-11-24: MySQL 실제 서버 연동 완료
- **DB 서버:** 준규 MySQL 서버 (172.30.1.60:3306)
- **연결 정보:** `.env` 파일에 `MYSQL_URL` 설정
- **테스트:** `DeepFake_DB/DB_test.py`로 연동 확인
- **Fallback:** MySQL 연결 실패 시 SQLite 사용 가능

### 2025-11-22: Firebase/MySQL 환경 변수 기반 연동
- **환경 변수:** `.env` 파일 기반 설정 (Git 제외)
- **Firebase:** 서비스 계정 키 경로 및 RTDB URL 분리
- **MySQL:** 선택적 연동 (기본값 SQLite → 준규 서버로 변경)
- **테스트:** `DeepFake_DB/DB_test.py`로 연동 확인
- **보안:** `secrets/`, `.env`, `*.json` Git 추적 제외

### 2025-11-21: YouTube 다운로드 라이브러리 변경
- **이전:** pytube (YouTube API 변경에 취약, 자주 오류 발생)
- **변경:** yt-dlp (안정적이고 지속적으로 업데이트됨)
- **영향받는 파일:**
  - `requirements.txt`: pytube → yt-dlp
  - `app/services/youtube.py`: 전체 구현 변경

### 프론트엔드 API 통신 수정
- **문제:** YouTube 탐지 API 호출 시 422 에러 (필드 검증 실패)
- **원인:** 백엔드는 Form 데이터를 기대하는데 프론트엔드가 JSON으로 전송
- **수정:** `deepfake_web/services/backend_api.py`의 `post_detect_youtube()` 함수
  - `requests.post(..., json=payload)` → `requests.post(..., data=data)`
- **결과:** YouTube 링크 탐지 정상 작동 (Status 200)

### 테스트 완료
- ✅ YouTube 영상 다운로드 (`yt-dlp` 사용)
- ✅ 딥페이크 탐지 API 호출 (Form 데이터 전송)
- ✅ 결과 반환 및 DB 저장
- ✅ Firebase 로그 기록
- ✅ SQLite 임시 테스트
- ✅ 환경 변수 기반 설정

### 설치 방법
기존 환경에서 업데이트하려면:
```bash
# conda 환경 활성화
conda activate deepfake_backend_env

# 필수 패키지 설치/업데이트
pip install -r requirements.txt

# 서버 재시작
uvicorn app.main:app --reload
```

### 필수 패키지
- `yt-dlp`: YouTube 다운로드
- `mediapipe`, `opencv-python`: 랜드마크 추출
- `numpy==1.24.3`: mediapipe 호환성
- `firebase-admin`: Firebase RTDB 연동
- `python-dotenv`: 환경 변수 로드
- `bcrypt`: 비밀번호 해싱
- `sqlalchemy`, `pymysql`: DB 연동
