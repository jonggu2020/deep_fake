# 🕵️ Deepfake Detection System

종구님의 고급 딥페이크 탐지 모델(XGBoost + RNN AE + MultiModal AE 앙상블)을 **FastAPI 백엔드 + Streamlit 프론트엔드**로 통합한 시스템입니다.

---

## 🔧 프로젝트 문제 발생과 해결 방법

| 문제 | 설명 | 해결 방법 |
|------|------|---------|
| **얼굴 인식 실패** | 영상에서 얼굴을 못찾아 분석 불가 | **분석 구간 설정**으로 명확한 얼굴 프레임만 분석 |
| **데이터 불일치** | 로그인 후 사용자 ID 미매칭 | API 응답 필드 수정 (user_id → id) |
| **YouTube 422 오류** | 데이터 형식 불일치로 분석 불가 | 요청 형식을 Form에서 JSON으로 변경 |
| **dlib 경로 오류** | 한글 경로에서 dlib 파일 인식 불가 | C:\temp_dlib\ 영문 경로로 폴백 설정 |
| **인코딩 오류** | 이모지 출력 시 UnicodeEncodeError | 모든 print 문에서 이모지 제거 |

---

## 🚀 빠른 시작 (권장)

### ✅ 원클릭 실행

```bash
python start.py
```

**자동 실행 순서:**
1. ✅ FastAPI 백엔드 (포트 8000)
2. ✅ Streamlit 프론트엔드 (포트 8502)
3. ✅ ngrok 터널링 (외부 공개)

### 📍 접속 URL

실행 후 터미널에 표시되는 ngrok URL로 **전 세계 어디서나** 접속 가능합니다!

```
외부 공개 URL: https://xxxxx.ngrok-free.app
로컬 Streamlit: http://localhost:8502
API 문서: http://localhost:8000/docs
```

### 🔐 로그인 정보

```
Email: 4comma3@naver.com
Password: test123
```

---

## ⚙️ 환경 설정

### 1. Python 환경 생성

```bash
conda create -n deepfake_backend_env python=3.10
conda activate deepfake_backend_env
pip install -r requirements.txt
```

### 2. MySQL 설정 (필수)

`.env` 파일을 프로젝트 루트에 생성:

```env
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db
```

MySQL 데이터베이스 생성:
```sql
CREATE DATABASE deepfake_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'deepfake'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepfake_db.* TO 'deepfake'@'localhost';
FLUSH PRIVILEGES;
```

### 3. ngrok 설정 (필수)

1. https://ngrok.com/download 에서 다운로드
2. `C:\ngrok\ngrok.exe`에 저장
3. 다른 경로면 `start.py` 파일 상단의 `NGROK_PATH` 수정

### 4. dlib 파일 설정 (한글 경로 문제 해결)

```bash
# C:\temp_dlib\ 폴더 생성
mkdir C:\temp_dlib

# shape_predictor_68_face_landmarks.dat 파일을 C:\temp_dlib\로 복사
copy "app\models_jonggu\models\shape_predictor_68_face_landmarks.dat" "C:\temp_dlib\"
```

---

## 🎯 주요 기능

### ✅ 사용자 관리
- 이메일 회원가입/로그인
- bcrypt 암호 해싱
- MySQL + Firebase 저장

### ✅ 비디오 분석
- **파일 업로드**: 로컬 비디오 직접 분석
- **YouTube 분석**: URL 자동 다운로드 후 분석
  - 시간 범위 선택 기능 (예: 0~15초만 분석)
  - 얼굴 특징 랜드마크 시각화
- **종구님 모델**: XGBoost + RNN AE + MultiModal AE 앙상블
  - 음성 분석 (Whisper + librosa)
  - 얼굴 특징 추출 (dlib 68-point)
  - 실시간 특징 시각화

---

## 🔧 트러블슈팅

### ngrok을 찾을 수 없습니다

```
[ERROR] ngrok 파일을 찾을 수 없습니다
```

**해결:**
1. https://ngrok.com/download 에서 다운로드
2. `C:\ngrok\ngrok.exe`에 저장
3. 또는 `start.py`의 `NGROK_PATH` 수정

### MySQL 연결 실패

```
CRITICAL: MYSQL_URL 환경변수를 찾을 수 없습니다!
```

**해결:**
1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. `MYSQL_URL=mysql+pymysql://...` 형식 확인
3. MySQL 서버가 실행 중인지 확인

### Streamlit이 바로 종료됨

**원인:** FastAPI 백엔드가 먼저 실행되지 않았거나 포트 충돌

**해결:**
```powershell
# 모든 Python 프로세스 종료
taskkill /F /IM python.exe

# 다시 실행
python start.py
```

### UnicodeEncodeError (cp949 인코딩 오류)

**원인:** Windows PowerShell이 특정 문자를 인코딩하지 못함

**해결:** 이미 모든 print 문에서 이모지 제거됨 (최신 버전 사용)

---

## 📁 프로젝트 구조

```
deepfake_backend/
├── app/
│   ├── main.py                          # FastAPI 진입점
│   ├── database.py                      # MySQL 설정
│   ├── routers/
│   │   ├── auth.py                      # 로그인/회원가입
│   │   └── detect.py                    # 분석 API
│   ├── models/
│   │   ├── user.py                      # 사용자 테이블
│   │   └── video.py                     # 비디오 테이블
│   ├── models_jonggu/
│   │   ├── deepfake_detector_webapp.py  # 종구님 Streamlit ⭐
│   │   └── models/
│   │       ├── HQ/                      # 고품질 모델
│   │       └── shape_predictor...dat    # dlib 얼굴 인식
│   └── services/
│       ├── jonggu_deepfake.py           # 종구님 모델 통합
│       ├── youtube.py                   # YouTube 다운로드
│       └── firebase_logger.py           # Firebase 로깅
├── uploads/                             # 업로드 파일 저장
├── .env                                 # 환경 변수 (Git 제외)
├── requirements.txt                     # 패키지 목록
├── start.py                             # 통합 실행 스크립트 ⭐
└── README.md                            # 이 파일
```

---

## 📦 주요 패키지

- **fastapi**: 0.123.5 (백엔드)
- **streamlit**: 1.51.0 (프론트엔드)
- **sqlalchemy**: 2.0.44 (ORM)
- **bcrypt**: 4.0.1 (암호 해싱)
- **torch**: 2.9.1 (딥러닝)
- **xgboost**: 3.1.2 (앙상블)
- **librosa**: 0.11.0 (음성 처리)
- **opencv**: 4.12.0 (비디오 처리)

전체 목록: `requirements.txt` 참고

---

## 🔄 동작 흐름

```
사용자 접속 (ngrok URL)
    ↓
Streamlit UI (8502)
    ↓
로그인 (선택사항)
    ↓
비디오 분석 요청
    ↓
FastAPI 백엔드 (8000)
    ↓
종구님 모델 실행
(XGBoost + RNN AE + MultiModal AE)
    ↓
MySQL DB 저장
    ↓
결과 반환 (FAKE/REAL 판정)
```

---

## 📈 성능 정보

| 항목 | 설명 |
|------|------|
| 평균 분석 시간 | 10~30초 (비디오 길이에 따라 다름) |
| 최대 파일 크기 | 200MB (설정 가능) |
| 지원 포맷 | MP4, AVI, MKV, MOV |
| 외부 접근 | ngrok 통해 전 세계 공개 |

---

## ✅ 배포 체크리스트

- [ ] Python 환경 생성 및 패키지 설치
- [ ] `.env` 파일 생성 (MySQL URL 설정)
- [ ] MySQL 데이터베이스 생성
- [ ] ngrok 다운로드 및 설치 (C:\ngrok\)
- [ ] dlib 파일을 C:\temp_dlib\로 복사
- [ ] `python start.py` 실행
- [ ] 브라우저에서 ngrok URL 접속 확인
- [ ] 테스트 계정으로 로그인 확인
- [ ] YouTube 분석 테스트

---

**🚀 준비됐으면 `python start.py` 실행하세요!**

## ⚙️ 환경 설정

### 1. MySQL 설정 (필수)

`.env` 파일을 프로젝트 루트에 생성:

```env
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db
```

MySQL 기본 설정:
```sql
CREATE DATABASE deepfake_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'deepfake'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepfake_db.* TO 'deepfake'@'localhost';
FLUSH PRIVILEGES;
```

### 2. Firebase 설정 (선택사항)

`.env` 파일에 추가:

```env
FIREBASE_CREDENTIALS=secrets/your-firebase-key.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/
```

---

## 🎯 주요 기능

### ✅ 사용자 관리
- 이메일 회원가입
- 이메일/비밀번호 로그인
- bcrypt 암호 해싱
- MySQL + Firebase 저장

### ✅ 비디오 분석
- **파일 업로드**: 로컬 비디오 직접 분석
- **YouTube 분석**: URL 자동 다운로드 후 분석 (로그인 필수)
- **종구님 모델**: XGBoost + RNN AE + MultiModal AE 앙상블
  - 음성 분석 (Whisper + librosa)
  - 얼굴 특징 추출 (dlib 68-point)
  - 실시간 특징 시각화

### ✅ 편의 기능
- 🔧 ngrok 자동 터널링 (외부 접근)
- 📊 Swagger UI (http://localhost:8000/docs)
- 📝 실시간 진행도 표시
- 💾 모든 결과 DB 저장

---

## 🔐 로그인 기능

### Streamlit 앱에서
1. **왼쪽 Sidebar** → **🔐 로그인** 섹션
2. **로그인 탭**: 기존 사용자 로그인
3. **회원가입 탭**: 새 계정 생성
4. 로그인 후 모든 기능 이용 가능

### 테스트 계정
- **이메일**: 4comma3@naver.com
- **비밀번호**: test123

---

## 🎬 분석 방법

### 파일 업로드로 분석
1. Streamlit 앱 열기 (http://localhost:8502)
2. **📥 입력 방식 선택** → **파일 업로드**
3. 비디오 파일 선택 (MP4, AVI, MKV, MOV)
4. **분석 시작** 버튼 클릭
5. 결과 확인 (FAKE/REAL 판정)

### YouTube로 분석
1. **📥 입력 방식 선택** → **YouTube 링크**
2. YouTube URL 입력
3. **YouTube 비디오 분석 시작** 버튼 클릭
4. 결과 확인

### 민감도 조절
- **Sensitivity (K)** 슬라이더 조정
- K=1.0: 엄격한 판정
- K=2.0: 기본값 (권장)
- K=3.0+: 관대한 판정

---

## 📊 분석 결과

```json
{
  "video_id": 5,
  "fake_probability": 87.5,
  "is_fake": true,
  "input_sharpness": 156.8,
  "scores": {
    "xgboost": 0.92,
    "rnn_ae": 0.81,
    "multimodal_ae": 0.85
  }
}
```

- **fake_probability**: 딥페이크 확률 (0~100%)
- **is_fake**: true (FAKE), false (REAL)
- **scores**: 각 모델의 개별 점수

---

## 🔗 API 사용 예시

### cURL
```bash
curl -X POST "http://localhost:8000/detect/jonggu-model" \
  -F "file=@video.mp4" \
  -F "user_id=1" \
  -F "sensitivity_k=2.0"
```

### Python
```python
import requests

with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/jonggu-model',
        files={'file': f},
        data={'user_id': 1, 'sensitivity_k': 2.0}
    )
    print(response.json())
```

### YouTube API
```python
import requests

response = requests.post(
    'http://localhost:8000/detect/youtube',
    json={
        'url': 'https://www.youtube.com/watch?v=...',
        'user_id': 1,
        'sensitivity_k': 2.0
    }
)
print(response.json())
```

**전체 API 문서**: http://localhost:8000/docs

---

## 🔧 트러블슈팅

### 포트 충돌
```
error while attempting to bind on address
```
**해결:**
```powershell
netstat -ano | Select-String "8000|8502"
taskkill /F /PID [PID번호]
```

### MySQL 연결 실패
```
CRITICAL: MYSQL_URL 환경변수를 찾을 수 없습니다!
```
**확인:**
1. `.env` 파일이 프로젝트 루트에 있는가?
2. `MYSQL_URL=mysql+pymysql://...` 형식이 맞는가?
3. MySQL 서버 실행 중인가?

### ngrok 설정 오류
```
ngrok.exe를 찾을 수 없습니다
```
**해결:**
1. https://ngrok.com/download 다운로드
2. `start.py` 수정:
```python
NGROK_PATH = r"C:\경로\to\ngrok.exe"
```

---

## 📁 프로젝트 구조

```
deepfake_backend/
├── app/
│   ├── main.py                          # FastAPI 진입점
│   ├── database.py                      # MySQL 설정
│   ├── routers/
│   │   ├── auth.py                      # 로그인/회원가입
│   │   └── detect.py                    # 분석 API
│   ├── models/
│   │   ├── user.py                      # 사용자 테이블
│   │   └── video.py                     # 비디오 테이블
│   ├── models_jonggu/
│   │   ├── deepfake_detector_webapp.py  # 종구님 Streamlit ⭐
│   │   └── models/
│   │       ├── HQ/                      # 고품질 모델
│   │       └── LQ/                      # 저품질 모델
│   └── services/
│       ├── jonggu_deepfake.py           # 종구님 모델 통합
│       ├── youtube.py                   # YouTube 다운로드
│       └── firebase_logger.py           # Firebase 로깅
├── uploads/                             # 업로드 파일 저장
├── .env                                 # 환경 변수 (Git 제외)
├── requirements.txt                     # 패키지 목록
├── start.py                             # 통합 실행 스크립트
├── start.bat                            # Windows 배치 파일
└── README.md                            # 이 파일
```

---

## 📦 주요 패키지

- **fastapi**: 0.123.5 (백엔드)
- **streamlit**: 1.51.0 (프론트엔드)
- **sqlalchemy**: 2.0.44 (ORM)
- **bcrypt**: 4.0.1 (암호 해싱)
- **torch**: 2.9.1 (딥러닝)
- **xgboost**: 3.1.2 (앙상블)
- **librosa**: 0.11.0 (음성 처리)
- **opencv**: 4.12.0 (비디오 처리)

전체 목록: `requirements.txt` 참고

---

## 🔄 동작 흐름

```
┌─────────────────────────┐
│   사용자 (Streamlit)     │
│ http://localhost:8502   │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────────┐
    │ 로그인 (선택사항)   │
    │ /auth/login       │
    └───────────┬───────┘
                │
    ┌───────────▼──────────────┐
    │ 분석 요청                  │
    │ • 파일 업로드              │
    │ • YouTube URL             │
    └───────────┬──────────────┘
                │
    ┌───────────▼─────────────────────┐
    │    FastAPI 백엔드 (8000)         │
    │                                 │
    │  ┌──────────────────────────┐   │
    │  │ 종구님 모델 실행          │   │
    │  │ (XGBoost + RNN AE)       │   │
    │  └──────────────────────────┘   │
    └───────────┬─────────────────────┘
                │
    ┌───────────▼──────────────┐
    │ MySQL DB 저장           │
    │ + Firebase 로깅          │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────┐
    │ 결과 반환                  │
    │ (FAKE/REAL 판정)         │
    └──────────────────────────┘
```

---

## 📈 성능 정보

| 항목 | 설명 |
|------|------|
| 평균 분석 시간 | 10~30초 (모델 크기 및 영상 길이 따라 다름) |
| 최대 파일 크기 | 200MB (설정 가능) |
| 지원 포맷 | MP4, AVI, MKV, MOV |
| 외부 접근 | ngrok 통해 인터넷 공개 |

---

## 🎓 기술 스택

### 백엔드
- FastAPI + Uvicorn
- SQLAlchemy ORM
- MySQL + Firebase Realtime DB

### 프론트엔드
- Streamlit 1.51.0

### AI/ML
- PyTorch 2.9.1
- XGBoost 3.1.2
- librosa (음성)
- OpenCV (영상)
- MediaPipe (얼굴 감지)
- dlib (얼굴 특징)

### 배포
- ngrok (외부 공개)
- Conda (환경 관리)

---

## ✅ 체크리스트

배포 전 확인:
- [ ] `.env` 파일 생성 (MySQL URL 설정)
- [ ] MySQL 데이터베이스 생성
- [ ] `python start.py` 실행 확인
- [ ] http://localhost:8502 접속 확인
- [ ] 테스트 계정으로 로그인 확인
- [ ] YouTube 분석 테스트
- [ ] 외부 ngrok URL 확인

---

## 📧 요약

이 시스템은:
1. **Streamlit 앱** (8502)을 메인 UI로 사용
2. **FastAPI 백엔드** (8000)에서 분석 처리
3. **종구님 모델**로 고정밀 딥페이크 탐지
4. **MySQL + Firebase**에 결과 저장
5. **ngrok**으로 외부 접근 가능

---

**🚀 준비됐으면 `python start.py` 실행하세요!**
