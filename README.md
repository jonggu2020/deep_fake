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
- Pytube
- ngrok

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
- `.env.example` 복사 후 `.env` 작성 (MySQL/Firebase 경로 설정)
- `secrets/firebase-service-account.json` 키 파일 배치 (Git 추적 제외)
- 필요 시 `MYSQL_URL` 환경변수로 MySQL 활성화 (미설정 시 SQLite)
- 업로드/유튜브 탐지 후 Firebase 로그 자동 기록 (키 없으면 건너뜀)