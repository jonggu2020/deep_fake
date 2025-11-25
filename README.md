# 🎭 Deepfake Detection System

딥페이크 탐지 프로젝트의 백엔드 + 프론트엔드 통합 시스템입니다.  
**한 번의 명령**으로 모든 서버를 실행할 수 있습니다.

---

## 🚀 빠른 시작 (3단계)

### 1️⃣ 환경 설치
```bash
# Conda 환경 생성
conda create -n deepfake_backend_env python=3.10
conda activate deepfake_backend_env

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 데이터베이스 설정

#### MySQL 설정 (필수)
1. MySQL Workbench에서 데이터베이스 생성:
```sql
CREATE DATABASE deepfake_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'deepfake'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepfake_db.* TO 'deepfake'@'localhost';
FLUSH PRIVILEGES;
```

2. `.env` 파일 생성 (프로젝트 루트에):
```env
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db
```

#### Firebase 설정 (선택)
1. Firebase Console에서 서비스 계정 키 다운로드
2. `secrets/` 폴더에 JSON 파일 저장
3. `.env`에 추가:
```env
FIREBASE_CREDENTIALS=secrets/your-firebase-key.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/
```

### 3️⃣ 실행
```bash
# Windows
start.bat

# 또는
python start.py
```

**자동으로 실행되는 것:**
- ✅ 포트 정리 (8000, 8501, 4040)
- ✅ FastAPI 백엔드 서버 (http://localhost:8000)
- ✅ Streamlit 프론트엔드 (http://localhost:8501)
- ✅ ngrok 터널링 (외부 접속용 HTTPS URL)

**종료:** 각 창에서 `Ctrl + C`

---

## 📋 주요 기능

### 1. 회원 관리
- 회원가입/로그인 (bcrypt 암호화)
- MySQL에 사용자 정보 저장
- Firebase에 사용자 동기화 (선택)

### 2. 딥페이크 탐지
- **파일 업로드**: 로컬 영상 파일 직접 업로드
- **YouTube 링크**: URL만으로 자동 다운로드 및 분석
- MediaPipe + OpenCV 기반 얼굴 분석
- Firebase에 탐지 결과 자동 로깅

### 3. 편의 기능
- 원클릭 실행 (포트 충돌 자동 해결)
- ngrok 자동 연동 (외부 접속 URL)
- Swagger UI 문서 자동 생성 (http://localhost:8000/docs)

---

## 🗂️ 프로젝트 구조

```
deepfake_backend/
├─ app/                          # FastAPI 백엔드
│  ├─ main.py                    # 서버 진입점
│  ├─ database.py                # MySQL 연결 설정
│  ├─ core/
│  │  └─ config.py              # 환경 변수 관리
│  ├─ routers/
│  │  ├─ auth.py                # 회원가입/로그인 API
│  │  └─ detect.py              # 딥페이크 탐지 API
│  ├─ models/
│  │  ├─ user.py                # User 테이블 모델
│  │  └─ video.py               # Video 테이블 모델
│  ├─ schemas/
│  │  ├─ user.py                # 요청/응답 스키마
│  │  └─ video.py
│  └─ services/
│     ├─ inference.py           # 딥페이크 탐지 로직
│     ├─ youtube.py             # YouTube 다운로드
│     └─ firebase_logger.py     # Firebase 로깅
│
├─ deepfake_web/                # Streamlit 프론트엔드
│  ├─ main.py                   # UI 진입점
│  ├─ views/
│  │  ├─ auth.py               # 로그인/회원가입 페이지
│  │  ├─ detect.py             # 탐지 페이지
│  │  └─ status.py             # 서버 상태 페이지
│  ├─ services/
│  │  ├─ backend_api.py        # FastAPI 클라이언트
│  │  └─ db.py                 # SQLite (로컬 히스토리)
│  └─ data/
│     └─ app.db                # Streamlit용 SQLite DB
│
├─ DeepFake_DB/                 # 데이터베이스 테스트
│  └─ DB_test.py               # MySQL/Firebase 연결 테스트
│
├─ uploads/                     # 업로드된 비디오 파일 저장
├─ secrets/                     # Firebase 키 (Git 제외)
├─ .env                         # 환경 변수 (Git 제외)
├─ requirements.txt             # Python 패키지 목록
├─ start.py                     # 통합 실행 스크립트
└─ start.bat                    # Windows 원클릭 실행
```

---

## 📡 API 문서

### 인증 API
| Method | Endpoint | 설명 | 요청 | 응답 |
|--------|----------|------|------|------|
| POST | `/auth/signup` | 회원가입 | `{"email": "user@example.com", "password": "pw123"}` | `{"id": 1, "email": "user@example.com", ...}` |
| POST | `/auth/login` | 로그인 | `{"email": "user@example.com", "password": "pw123"}` | `{"id": 1, "email": "user@example.com", ...}` |

### 탐지 API
| Method | Endpoint | 설명 | 요청 | 응답 |
|--------|----------|------|------|------|
| POST | `/detect/upload` | 파일 업로드 탐지 | `FormData(file, user_id)` | `{"video_id": 1, "result": "real/fake", ...}` |
| POST | `/detect/youtube` | YouTube 링크 탐지 | `{"user_id": 1, "youtube_url": "https://..."}` | `{"video_id": 1, "result": "real/fake", ...}` |

**Swagger UI:** http://localhost:8000/docs

---

## 🗄️ 데이터베이스 구조

### MySQL 테이블

#### users
```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### videos
```sql
CREATE TABLE videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    file_path VARCHAR(500),
    youtube_url VARCHAR(500),
    result VARCHAR(50) NOT NULL,
    confidence FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Firebase Realtime Database

```json
{
  "users": {
    "user_1": {
      "id": 1,
      "email": "user@example.com",
      "created_at": "2025-11-25T12:00:00",
      "last_login": "2025-11-25T12:30:00"
    }
  },
  "detection_logs": {
    "log_1": {
      "user_id": 1,
      "video_id": 1,
      "result": "fake",
      "confidence": 0.87,
      "timestamp": "2025-11-25T12:35:00"
    }
  }
}
```

---

## 🛠️ 기술 스택

### Backend
- **FastAPI** - 고성능 비동기 웹 프레임워크
- **Uvicorn** - ASGI 서버
- **SQLAlchemy** - ORM (MySQL 연동)
- **PyMySQL** - MySQL 드라이버
- **bcrypt 4.0.1** - 비밀번호 암호화

### Frontend
- **Streamlit** - 빠른 웹 UI 개발

### AI/ML
- **MediaPipe** - 얼굴 랜드마크 감지
- **OpenCV** - 비디오 처리
- **NumPy** - 수치 연산

### Database
- **MySQL** - 메인 데이터베이스
- **Firebase Realtime Database** - 로깅 및 동기화
- **SQLite** - Streamlit 로컬 히스토리

### Utils
- **yt-dlp** - YouTube 다운로드
- **ngrok** - 외부 접속 터널링
- **python-dotenv** - 환경 변수 관리

---

## ⚠️ 문제 해결

### 1. 포트 충돌
**증상:** `error while attempting to bind on address`

**해결:**
```bash
# PowerShell
netstat -ano | findstr :8000
taskkill /F /PID [PID번호]
```

### 2. MySQL 연결 실패
**증상:** `❌ CRITICAL: MYSQL_URL 환경변수가 없습니다!`

**해결:**
1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. `MYSQL_URL` 형식 확인:
   ```env
   MYSQL_URL=mysql+pymysql://username:password@127.0.0.1:3306/database_name
   ```
3. MySQL 서버 실행 여부 확인

### 3. bcrypt 오류
**증상:** `password cannot be longer than 72 bytes`

**해결:**
```bash
pip install "bcrypt==4.0.1" --force-reinstall
```

### 4. Firebase 저장 안됨
**증상:** 회원가입은 성공하지만 Firebase에 사용자 안 보임

**해결:**
1. `.env`에 Firebase 설정 확인
2. `secrets/` 폴더에 JSON 키 파일 존재 확인
3. Firebase Console에서 Database URL 확인

### 5. ngrok 경로 오류
**증상:** `ngrok.exe가 없습니다`

**해결:**
1. https://ngrok.com/download 에서 다운로드
2. `start.py` 파일 열어서 `NGROK_PATH` 수정:
   ```python
   NGROK_PATH = r"C:\경로\to\ngrok.exe"
   ```

---

## 🧪 테스트

### 데이터베이스 연결 테스트
```bash
conda activate deepfake_backend_env
python DeepFake_DB/DB_test.py
```

**예상 출력:**
```
✅ MySQL 연결 성공!
✅ Firebase 연결 성공!
📊 현재 사용자 수: 5
📊 현재 비디오 수: 12
```

### API 테스트
```bash
# 서버 실행 후
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123"}'
```

---

## 📦 패키지 목록

```txt
fastapi                 # 웹 프레임워크
uvicorn[standard]      # ASGI 서버
SQLAlchemy             # ORM
pymysql                # MySQL 드라이버
python-multipart       # 파일 업로드
pydantic[email]        # 데이터 검증
pydantic-settings      # 설정 관리
bcrypt==4.0.1          # 비밀번호 암호화 (중요: 버전 고정)
passlib[bcrypt]        # 암호화 헬퍼
yt-dlp                 # YouTube 다운로드
firebase-admin         # Firebase SDK
opencv-python          # 비디오 처리
mediapipe              # 얼굴 감지
numpy                  # 수치 연산
```

---

## 🔐 환경 변수 (.env)

```env
# MySQL (필수)
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db

# Firebase (선택)
FIREBASE_CREDENTIALS=secrets/your-firebase-key.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/
```

**⚠️ 주의:** `.env` 파일은 Git에 커밋하지 마세요! (`.gitignore`에 포함됨)

---

## 📝 개발 로그

### 주요 해결 사항
1. ✅ SQLite fallback 제거 - MySQL only
2. ✅ bcrypt 버전 문제 해결 (5.0.0 → 4.0.1)
3. ✅ Firebase 초기화 순서 개선
4. ✅ .env 로딩 강제 적용 (`override=True`)
5. ✅ 포트 자동 정리 기능 추가
6. ✅ 원클릭 실행 시스템 구축

---

## 👥 팀원 가이드

### 처음 시작하는 경우
1. 이 README의 "빠른 시작" 섹션 따라하기
2. MySQL 설정 필수 (.env 파일 작성)
3. `start.bat` 실행
4. http://localhost:8501 접속

### 개발 시
- 백엔드 코드 수정: `app/` 폴더
- 프론트엔드 코드 수정: `deepfake_web/` 폴더
- API 문서: http://localhost:8000/docs

### 커밋 전
- 테스트 파일 생성했으면 삭제
- `.env` 파일 커밋 금지
- `secrets/` 폴더 내용 커밋 금지

---

## 📞 문의
문제가 있으면 이슈 등록 또는 팀 채널에 문의하세요.
