# ?렚 Deepfake Detection System

?ν럹?댄겕 ?먯? ?꾨줈?앺듃??諛깆뿏??+ ?꾨줎?몄뿏???듯빀 ?쒖뒪?쒖엯?덈떎.  
**??踰덉쓽 紐낅졊**?쇰줈 紐⑤뱺 ?쒕쾭瑜??ㅽ뻾?????덉뒿?덈떎.

?넅 **v2.0 ?낅뜲?댄듃**: 醫낃뎄?섏쓽 怨좉툒 ?ν럹?댄겕 ?먯? 紐⑤뜽(XGBoost + RNN AE + MultiModal AE ?숈긽釉? ?듯빀 ?꾨즺!

---

## ?? 鍮좊Ⅸ ?쒖옉 (3?④퀎)

### 1截뤴깵 ?섍꼍 ?ㅼ튂
```bash
# Conda ?섍꼍 ?앹꽦
conda create -n deepfake_backend_env python=3.10
conda activate deepfake_backend_env

# ?⑦궎吏 ?ㅼ튂
pip install -r requirements.txt
```

### 2截뤴깵 ?곗씠?곕쿋?댁뒪 ?ㅼ젙

#### MySQL ?ㅼ젙 (?꾩닔)
1. MySQL Workbench?먯꽌 ?곗씠?곕쿋?댁뒪 ?앹꽦:
```sql
CREATE DATABASE deepfake_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'deepfake'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepfake_db.* TO 'deepfake'@'localhost';
FLUSH PRIVILEGES;
```

2. `.env` ?뚯씪 ?앹꽦 (?꾨줈?앺듃 猷⑦듃??:
```env
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db
```

#### Firebase ?ㅼ젙 (?좏깮)
1. Firebase Console?먯꽌 ?쒕퉬??怨꾩젙 ???ㅼ슫濡쒕뱶
2. `secrets/` ?대뜑??JSON ?뚯씪 ???3. `.env`??異붽?:
```env
FIREBASE_CREDENTIALS=secrets/your-firebase-key.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/
```

### 3截뤴깵 ?ㅽ뻾

#### 諛⑸쾿 1截뤴깵: Windows ?먰겢由??ㅽ뻾 (沅뚯옣)
```bash
start.bat
```

#### 諛⑸쾿 2截뤴깵: PowerShell/?곕???```bash
cd deepfake_backend
python start.py
```

#### 諛⑸쾿 3截뤴깵: ?섎룞 ?ㅽ뻾
```bash
# ?곕???1: FastAPI 諛깆뿏??cd deepfake_backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ?곕???2: Streamlit ?꾨줎?몄뿏??cd deepfake_backend
streamlit run deepfake_web/main.py
```

**?먮룞?쇰줈 ?ㅽ뻾?섎뒗 寃?**
- ???ы듃 ?뺣━ (8000, 8501, 4040)
- ??FastAPI 諛깆뿏???쒕쾭 (http://localhost:8000)
- ??Streamlit ?꾨줎?몄뿏??(http://localhost:8501)
- ??ngrok ?곕꼸留?(?몃? ?묒냽??HTTPS URL)

**醫낅즺:** 媛?李쎌뿉??`Ctrl + C`

---

## ?뱥 二쇱슂 湲곕뒫

### 1. ?뚯썝 愿由?- ?뚯썝媛??濡쒓렇??(bcrypt ?뷀샇??
- MySQL???ъ슜???뺣낫 ???- Firebase???ъ슜???숆린??(?좏깮)

### 2. ?ν럹?댄겕 ?먯?
- **湲곕낯 ?먯?**: `POST /detect/upload` - 濡쒖뺄 ?곸긽 ?뚯씪 吏곸젒 ?낅줈??- **YouTube ?먯?**: `POST /detect/youtube` - URL留뚯쑝濡??먮룞 ?ㅼ슫濡쒕뱶 諛?遺꾩꽍
- **怨좉툒 ?먯?** 狩? `POST /detect/jonggu-model` - 醫낃뎄??紐⑤뜽 (XGBoost + RNN AE + MultiModal AE)
  - ?뚯꽦 遺꾩꽍 (Whisper + librosa)
  - ?쇨뎬 ?뱀쭠 異붿텧 (dlib 68-point landmarks)
  - ?숈긽釉?紐⑤뜽 寃곌낵 諛??좊ː??- MediaPipe + OpenCV 湲곕컲 ?쇨뎬 遺꾩꽍
- Firebase???먯? 寃곌낵 ?먮룞 濡쒓퉭

### 3. ?몄쓽 湲곕뒫
- ?먰겢由??ㅽ뻾 (?ы듃 異⑸룎 ?먮룞 ?닿껐)
- ngrok ?먮룞 ?곕룞 (?몃? ?묒냽 URL)
- Swagger UI 臾몄꽌 ?먮룞 ?앹꽦 (http://localhost:8000/docs)

---

## ?뾺截??꾨줈?앺듃 援ъ“

```
deepfake_backend/
?쒋? app/                          # FastAPI 諛깆뿏???? ?쒋? main.py                    # ?쒕쾭 吏꾩엯???? ?쒋? database.py                # MySQL ?곌껐 ?ㅼ젙
?? ?쒋? core/
?? ?? ?붴? config.py              # ?섍꼍 蹂??愿由??? ?쒋? routers/
?? ?? ?쒋? auth.py                # ?뚯썝媛??濡쒓렇??API
?? ?? ?붴? detect.py              # ?ν럹?댄겕 ?먯? API
?? ?쒋? models/
?? ?? ?쒋? user.py                # User ?뚯씠釉?紐⑤뜽
?? ?? ?붴? video.py               # Video ?뚯씠釉?紐⑤뜽
?? ?쒋? schemas/
?? ?? ?쒋? user.py                # ?붿껌/?묐떟 ?ㅽ궎留??? ?? ?붴? video.py
?? ?쒋? models_jonggu/            # 醫낃뎄???ν럹?댄겕 ?먯? 紐⑤뜽 (XGBoost + ?숈긽釉?
?? ?? ?쒋? models/
?? ?? ?? ?쒋? HQ/                # 怨좏뭹吏??숈뒿 紐⑤뜽
?? ?? ?? ?붴? LQ/                # ??덉쭏 ?숈뒿 紐⑤뜽
?? ?? ?쒋? shape_predictor_68...dat  # dlib ?쇨뎬 ?쒕뱶留덊겕 媛먯?湲??? ?? ?붴? deepfake_detector_webapp.py # 醫낃뎄???먮낯 肄붾뱶
?? ?붴? services/
??    ?쒋? inference.py           # 湲곕낯 ?쒕뜡 ?먯? 濡쒖쭅
??    ?쒋? jonggu_deepfake.py     # 醫낃뎄??紐⑤뜽 ?쒕퉬??(NEW!)
??    ?쒋? youtube.py             # YouTube ?ㅼ슫濡쒕뱶
??    ?쒋? firebase_logger.py     # Firebase 濡쒓퉭
??    ?붴? landmark_extractor.py  # ?쇨뎬 ?쒕뱶留덊겕 異붿텧
???쒋? deepfake_web/                # Streamlit ?꾨줎?몄뿏???? ?쒋? main.py                   # UI 吏꾩엯???? ?쒋? views/
?? ?? ?쒋? auth.py               # 濡쒓렇???뚯썝媛???섏씠吏
?? ?? ?쒋? detect.py             # ?먯? ?섏씠吏
?? ?? ?붴? status.py             # ?쒕쾭 ?곹깭 ?섏씠吏
?? ?쒋? services/
?? ?? ?쒋? backend_api.py        # FastAPI ?대씪?댁뼵???? ?? ?붴? db.py                 # SQLite (濡쒖뺄 ?덉뒪?좊━)
?? ?붴? data/
??    ?붴? app.db                # Streamlit??SQLite DB
???쒋? DeepFake_DB/                 # ?곗씠?곕쿋?댁뒪 ?뚯뒪???? ?붴? DB_test.py               # MySQL/Firebase ?곌껐 ?뚯뒪?????쒋? uploads/                     # ?낅줈?쒕맂 鍮꾨뵒???뚯씪 ????쒋? secrets/                     # Firebase ??(Git ?쒖쇅)
?쒋? .env                         # ?섍꼍 蹂??(Git ?쒖쇅)
?쒋? requirements.txt             # Python ?⑦궎吏 紐⑸줉
?쒋? start.py                     # ?듯빀 ?ㅽ뻾 ?ㅽ겕由쏀듃
?붴? start.bat                    # Windows ?먰겢由??ㅽ뻾
```

---

## ?뱻 API 臾몄꽌

### ?몄쬆 API
| Method | Endpoint | ?ㅻ챸 | ?붿껌 | ?묐떟 |
|--------|----------|------|------|------|
| POST | `/auth/signup` | ?뚯썝媛??| `{"email": "user@example.com", "password": "pw123"}` | `{"id": 1, "email": "user@example.com", ...}` |
| POST | `/auth/login` | 濡쒓렇??| `{"email": "user@example.com", "password": "pw123"}` | `{"id": 1, "email": "user@example.com", ...}` |

### ?먯? API
| Method | Endpoint | ?ㅻ챸 | ?묐떟 |
|--------|----------|------|------|
| POST | `/detect/upload` | 湲곕낯 ?먯? (?뚯씪) | `{"video_id": 1, "fake_probability": 0.45, ...}` |
| POST | `/detect/youtube` | 湲곕낯 ?먯? (YouTube) | `{"video_id": 1, "fake_probability": 0.45, ...}` |
| POST | `/detect/jonggu-model` | **怨좉툒 ?먯?** (醫낃뎄??紐⑤뜽) | `{"video_id": 1, "fake_probability": 87.5, "is_fake": true, ...}` |
| GET | `/detect/landmark/{video_id}` | ?쒕뱶留덊겕 ?곸긽 諛섑솚 | ?곸긽 ?뚯씪 (MP4) |

**Swagger UI:** http://localhost:8000/docs

---

## ?뾼截??곗씠?곕쿋?댁뒪 援ъ“

### MySQL ?뚯씠釉?
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

## ?썱截?湲곗닠 ?ㅽ깮

### Backend
- **FastAPI** - 怨좎꽦??鍮꾨룞湲????꾨젅?꾩썙??- **Uvicorn** - ASGI ?쒕쾭
- **SQLAlchemy** - ORM (MySQL ?곕룞)
- **PyMySQL** - MySQL ?쒕씪?대쾭
- **bcrypt 4.0.1** - 鍮꾨?踰덊샇 ?뷀샇??
### Frontend
- **Streamlit** - 鍮좊Ⅸ ??UI 媛쒕컻

### AI/ML
- **MediaPipe** - ?쇨뎬 ?쒕뱶留덊겕 媛먯?
- **OpenCV** - 鍮꾨뵒??泥섎━
- **NumPy** - ?섏튂 ?곗궛

### Database
- **MySQL** - 硫붿씤 ?곗씠?곕쿋?댁뒪
- **Firebase Realtime Database** - 濡쒓퉭 諛??숆린??- **SQLite** - Streamlit 濡쒖뺄 ?덉뒪?좊━

### Utils
- **yt-dlp** - YouTube ?ㅼ슫濡쒕뱶
- **ngrok** - ?몃? ?묒냽 ?곕꼸留?- **python-dotenv** - ?섍꼍 蹂??愿由?
---

## ?좑툘 臾몄젣 ?닿껐

### 1. ?ы듃 異⑸룎
**利앹긽:** `error while attempting to bind on address`

**?닿껐:**
```bash
# PowerShell
netstat -ano | findstr :8000
taskkill /F /PID [PID踰덊샇]
```

### 2. MySQL ?곌껐 ?ㅽ뙣
**利앹긽:** `??CRITICAL: MYSQL_URL ?섍꼍蹂?섍? ?놁뒿?덈떎!`

**?닿껐:**
1. `.env` ?뚯씪???꾨줈?앺듃 猷⑦듃???덈뒗吏 ?뺤씤
2. `MYSQL_URL` ?뺤떇 ?뺤씤:
   ```env
   MYSQL_URL=mysql+pymysql://username:password@127.0.0.1:3306/database_name
   ```
3. MySQL ?쒕쾭 ?ㅽ뻾 ?щ? ?뺤씤

### 3. bcrypt ?ㅻ쪟
**利앹긽:** `password cannot be longer than 72 bytes`

**?닿껐:**
```bash
pip install "bcrypt==4.0.1" --force-reinstall
```

### 4. Firebase ????덈맖
**利앹긽:** ?뚯썝媛?낆? ?깃났?섏?留?Firebase???ъ슜????蹂댁엫

**?닿껐:**
1. `.env`??Firebase ?ㅼ젙 ?뺤씤
2. `secrets/` ?대뜑??JSON ???뚯씪 議댁옱 ?뺤씤
3. Firebase Console?먯꽌 Database URL ?뺤씤

### 5. ngrok 寃쎈줈 ?ㅻ쪟
**利앹긽:** `ngrok.exe媛 ?놁뒿?덈떎`

**?닿껐:**
1. https://ngrok.com/download ?먯꽌 ?ㅼ슫濡쒕뱶
2. `start.py` ?뚯씪 ?댁뼱??`NGROK_PATH` ?섏젙:
   ```python
   NGROK_PATH = r"C:\寃쎈줈\to\ngrok.exe"
   ```

---

## ?㎦ ?뚯뒪??
### ?곗씠?곕쿋?댁뒪 ?곌껐 ?뚯뒪??```bash
conda activate deepfake_backend_env
cd deepfake_backend
python -c "from app.database import engine; conn = engine.connect(); print('??MySQL ?곌껐 ?깃났!')"
```

### API ?뚯뒪??```bash
# ?쒕쾭 ?ㅽ뻾 ??curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123"}'
```

---

## ?벀 ?⑦궎吏 紐⑸줉

?먯꽭???댁슜? `requirements.txt` 李멸퀬

---

## ?뵍 ?섍꼍 蹂??(.env)

```env
# MySQL (?꾩닔)
MYSQL_URL=mysql+pymysql://deepfake:your_password@127.0.0.1:3306/deepfake_db

# Firebase (?좏깮)
FIREBASE_CREDENTIALS=secrets/your-firebase-key.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/
```

**?좑툘 二쇱쓽:** `.env` ?뚯씪? Git??而ㅻ컠?섏? 留덉꽭?? (`.gitignore`???ы븿??

---

## ?넅 醫낃뎄??紐⑤뜽 ?듯빀 媛?대뱶

### 怨좉툒 ?먯? ?붾뱶?ъ씤???ъ슜

#### cURL濡??뚯뒪??```bash
curl -X POST "http://localhost:8000/detect/jonggu-model" \
  -F "file=@video.mp4" \
  -F "user_id=1" \
  -F "sensitivity_k=2.0"
```

#### Python?쇰줈 ?ъ슜
```python
import httpx

with open('video.mp4', 'rb') as f:
    response = httpx.post(
        'http://localhost:8000/detect/jonggu-model',
        files={'file': f},
        data={'user_id': 1, 'sensitivity_k': 2.0}
    )
    print(response.json())
```

#### ?묐떟 ?덉떆
```json
{
  "video_id": 5,
  "fake_probability": 87.5,
  "is_fake": true,
  "analysis_range": {"start": 2.3, "end": 7.3},
  "input_sharpness": 156.8,
  "sensitivity_factor": 2.34,
  "scores": {
    "xgboost": 0.92,
    "rnn_ae": 0.81,
    "tabular_ae": 0.88,
    "multimodal_ae": 0.85
  }
}
```

### 誘쇨컧??議곗젙 (sensitivity_k)
- **k=1.0**: ??? 誘쇨컧??(?곴레?곸씤 ?먯?)
- **k=2.0**: 湲곕낯媛?(洹좏삎?≫엺 ?먯?)
- **k=3.0+**: ?믪? 誘쇨컧??(蹂댁닔?곸씤 ?먯?)

---

## ?뱷 媛쒕컻 濡쒓렇

### v2.0 (2025-12-04)
- ??醫낃뎄??怨좉툒 ?ν럹?댄겕 ?먯? 紐⑤뜽 ?듯빀
- ??`/detect/jonggu-model` ?붾뱶?ъ씤??異붽?
- ??XGBoost + RNN AE + MultiModal AE ?숈긽釉?- ???뚯꽦 遺꾩꽍 (Whisper + librosa) ?듯빀
- ??dlib ?쇨뎬 ?쒕뱶留덊겕 ?듯빀
- ??誘쇨컧??議곗젙 湲곕뒫 異붽?
- ??遺덊븘?뷀븳 ?뚯씪 諛??대뜑 ?뺣━

### v1.0 珥덇린 援ъ텞
1. ??SQLite fallback ?쒓굅 - MySQL only
2. ??bcrypt 踰꾩쟾 臾몄젣 ?닿껐 (5.0.0 ??4.0.1)
3. ??Firebase 珥덇린???쒖꽌 媛쒖꽑
4. ??.env 濡쒕뵫 媛뺤젣 ?곸슜 (`override=True`)
5. ???ы듃 ?먮룞 ?뺣━ 湲곕뒫 異붽?
6. ???먰겢由??ㅽ뻾 ?쒖뒪??援ъ텞

---

## ?뫁 ???媛?대뱶

### 泥섏쓬 ?쒖옉?섎뒗 寃쎌슦
1. ??README??"鍮좊Ⅸ ?쒖옉" ?뱀뀡 ?곕씪?섍린
2. MySQL ?ㅼ젙 ?꾩닔 (.env ?뚯씪 ?묒꽦)
3. `start.bat` ?먮뒗 `python start.py` ?ㅽ뻾
4. http://localhost:8501 ?묒냽 (?꾨줎?몄뿏??
5. http://localhost:8000/docs ?묒냽 (API 臾몄꽌)

### 媛쒕컻 ??- **諛깆뿏??肄붾뱶**: `app/` ?대뜑
  - ?쇱슦?? `app/routers/`
  - ?쒕퉬?? `app/services/`
  - 紐⑤뜽: `app/models/`
- **?꾨줎?몄뿏??肄붾뱶**: `deepfake_web/` ?대뜑
- **醫낃뎄??紐⑤뜽**: `app/models_jonggu/` ?대뜑 (?섏젙 湲덉?)
- **API 臾몄꽌**: http://localhost:8000/docs (?먮룞 ?앹꽦)

### 而ㅻ컠 ??泥댄겕由ъ뒪??- [ ] ?뚯뒪???뚯씪 ?앹꽦?덉쑝硫???젣
- [ ] `.env` ?뚯씪 而ㅻ컠 湲덉? (?대? `.gitignore`??異붽???
- [ ] `secrets/` ?대뜑 ?댁슜 而ㅻ컠 湲덉?
- [ ] `uploads/` ?대뜑???꾩떆 ?곸긽 ?뚯씪 ?뺣━
- [ ] 遺덊븘?뷀븳 `__pycache__` ?붾젆?좊━ ?뺤씤

### 醫낃뎄??紐⑤뜽 愿??- **紐⑤뜽 ?뚯씪 寃쎈줈**: `app/models_jonggu/models/HQ/` (怨좏뭹吏? 諛?`LQ/` (??덉쭏)
- **?숈뒿??紐⑤뜽**: XGBoost, RNN AE, Tabular AE, MultiModal AE
- **?섏〈??*: torch, xgboost, librosa, dlib (?먮룞 ?ㅼ튂)
- **紐⑤뜽 蹂寃?湲덉?**: `jonggu_deepfake.py`??紐⑤뜽 濡쒕뱶 遺遺??섏젙?섏? 留덉꽭??
---

## ?뱸 臾몄쓽
臾몄젣媛 ?덉쑝硫??댁뒋 ?깅줉 ?먮뒗 ? 梨꾨꼸??臾몄쓽?섏꽭??
