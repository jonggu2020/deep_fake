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
│     └─ youtube.py
├─ uploads/
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
- inference.py: 분석 로직(현재 랜덤)
- youtube.py: 유튜브 다운로드

## 5. 개발 환경 세팅
### 1) 가상환경
```
conda create -n deepfake_backend_env python=3.10
conda activate deepfake_backend_env
```

### 2) 라이브러리 설치
```
pip install -r requirements.txt
```

## 6. 서버 실행
```
(첫번째 터미널)
uvicorn app.main:app --reload
```

## 7. 외부 접속 (ngrok)
```
(두번째 터미널)
./ngrok http 8000
```

Swagger:
```
https://xxxx.ngrok-free.app/docs
```

## 8. 제공 API
- GET /
- POST /auth/signup
- POST /auth/login
- POST /detect/upload
- POST /detect/youtube

## 9. 향후 확장
- 실제 모델 연결
- MySQL/Firebase 연동
- JWT 인증
- 로깅/예외 처리 추가

## 10. 통합 가이드 요약
프론트(HOTTI) + Firebase 로그 저장 + 선택적 MySQL 사용을 위한 상세 절차는 `INTEGRATION_GUIDE.md` 참고.

핵심 요약:
- `.env.example` 복사 후 `.env` 작성 (MySQL/Firebase 경로 설정)
- `secrets/firebase-service-account.json` 키 파일 배치 (Git 추적 제외)
- 필요 시 `MYSQL_URL` 환경변수로 MySQL 활성화 (미설정 시 SQLite)
- 업로드/유튜브 탐지 후 Firebase 로그 자동 기록 (키 없으면 건너뜀)