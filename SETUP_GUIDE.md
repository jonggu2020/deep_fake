# 🚀 실행 가이드 (빠른 참고)

## 🎯 3가지 실행 방법

### 방법 1️⃣: Windows 원클릭 실행 (가장 간단! 권장)
```bash
start.bat
```
**이것이 제일 쉬워요!** Windows 탐색기에서 `start.bat`을 더블클릭하면 됩니다.

---

### 방법 2️⃣: 터미널에서 실행 (PowerShell/CMD)
```bash
cd deepfake_backend
python start.py
```

---

### 방법 3️⃣: 수동 실행 (각각 따로 실행)

#### 터미널 1: 백엔드 서버
```bash
cd deepfake_backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 터미널 2: 프론트엔드 서버
```bash
cd deepfake_backend
streamlit run deepfake_web/main.py
```

---

## 🌐 접속 URL

실행 후 다음 주소들을 브라우저에서 열어보세요:

| 항목 | URL | 설명 |
|------|-----|------|
| **프론트엔드** | http://localhost:8501 | 딥페이크 탐지 UI (Streamlit) |
| **API 문서** | http://localhost:8000/docs | Swagger UI - 모든 API 테스트 가능 |
| **백엔드 (정상 여부)** | http://localhost:8000/ | 백엔드 상태 확인 |

---

## 📝 처음 설정하기

### Step 1: 필요한 것들
- Python 3.10 이상
- MySQL 서버 (로컬 또는 원격)
- Conda 또는 pip

### Step 2: 환경 설정
```bash
# Conda 환경 생성
conda create -n deepfake_backend_env python=3.10
conda activate deepfake_backend_env

# 패키지 설치
pip install -r requirements.txt
```

### Step 3: `.env` 파일 생성

프로젝트 루트 (`deepfake_backend/` 폴더)에 `.env` 파일을 만들고 다음을 입력:

```env
# MySQL 설정 (필수)
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db

# Firebase 설정 (선택 - 로그를 Firebase에 저장하려면)
FIREBASE_CREDENTIALS=secrets/your-firebase-key.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/
```

**MySQL 설정 방법:**
1. MySQL을 설치하고 실행
2. MySQL Workbench에서 다음 SQL 실행:
```sql
CREATE DATABASE deepfake_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'deepfake'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepfake_db.* TO 'deepfake'@'localhost';
FLUSH PRIVILEGES;
```

### Step 4: 실행!
```bash
# Windows
start.bat

# 또는
python start.py
```

---

## 📊 API 활용 예시

### 고급 탐지 (종구님 모델) 사용

#### cURL로 테스트
```bash
curl -X POST "http://localhost:8000/detect/jonggu-model" \
  -F "file=@video.mp4" \
  -F "user_id=1" \
  -F "sensitivity_k=2.0"
```

#### Python으로 사용
```python
import httpx

with open('video.mp4', 'rb') as f:
    response = httpx.post(
        'http://localhost:8000/detect/jonggu-model',
        files={'file': f},
        data={'user_id': 1, 'sensitivity_k': 2.0}
    )
    result = response.json()
    print(f"딥페이크 확률: {result['fake_probability']}%")
    print(f"판정: {'딥페이크' if result['is_fake'] else '정상 영상'}")
```

---

## ⚠️ 자주 묻는 질문

### Q1: 포트 충돌 에러가 나요
**A:** PowerShell에서 다음을 실행:
```bash
netstat -ano | findstr :8000
taskkill /F /PID [PID번호]
```

### Q2: MySQL 연결 실패
**A:** 다음을 확인하세요:
- [ ] MySQL 서버가 실행 중인가?
- [ ] `.env` 파일이 프로젝트 루트에 있는가?
- [ ] `MYSQL_URL`의 비밀번호가 맞는가?
- [ ] `deepfake` 사용자가 `deepfake_db` DB에 권한이 있는가?

### Q3: 패키지 설치 에러
**A:** 다음을 시도:
```bash
pip install -r requirements.txt --no-cache-dir
```

### Q4: bcrypt 에러 (`password cannot be longer than 72 bytes`)
**A:** bcrypt를 재설치:
```bash
pip install "bcrypt==4.0.1" --force-reinstall
```

---

## 🎯 각 엔드포인트 설명

### 회원 관리
- `POST /auth/signup` - 회원가입
- `POST /auth/login` - 로그인

### 탐지 API
- `POST /detect/upload` - 파일 업로드 탐지 (기본 모델)
- `POST /detect/youtube` - YouTube 링크 탐지 (기본 모델)
- **`POST /detect/jonggu-model`** - 종구님 고급 모델 탐지 ⭐
- `GET /detect/landmark/{video_id}` - 랜드마크 영상 조회

---

## 📚 더 자세한 정보

자세한 내용은 `README.md`를 참고하세요!

---

## 💡 팁

1. **프론트엔드와 백엔드가 분리되어 실행됩니다:**
   - 프론트엔드: http://localhost:8501 (Streamlit)
   - 백엔드: http://localhost:8000 (FastAPI)

2. **API 문서가 자동 생성됩니다:**
   - http://localhost:8000/docs에서 모든 API를 테스트할 수 있습니다

3. **로그 저장:**
   - MySQL에 모든 탐지 결과가 저장됩니다
   - Firebase를 설정하면 추가로 로깅됩니다

4. **민감도 조정:**
   - `sensitivity_k=2.0` (기본값, 균형잡힘)
   - `sensitivity_k=1.0` (낮은 민감도, 적극적 탐지)
   - `sensitivity_k=3.0+` (높은 민감도, 보수적 탐지)
