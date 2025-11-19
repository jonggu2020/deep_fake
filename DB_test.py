import firebase_admin
from firebase_admin import credentials, db
from sqlalchemy import create_engine, text
from passlib.context import CryptContext

# --- 1. 암호화 설정 ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- 2. MySQL 연결 설정 ---
# (DB 정보는 팀원과 공유된 정보로 변경)
MYSQL_DATABASE_URL = "mysql+pymysql://root:(mysql password)@localhost/firebase_db_tset" 
engine = create_engine(MYSQL_DATABASE_URL)

# --- 3. Firebase 초기화 ---
if not firebase_admin._apps:
    cred_path = "C:/deep_fake_LJK/(firebase key)" 
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sw-deepfake-project-default-rtdb.firebaseio.com/'
    })

# --- 4. (가짜) 회원가입 함수 ---
def simulate_register(email, password):
    hashed_password = pwd_context.hash(password)
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO users (email, hashed_password) VALUES (:email, :hashed_password)"
        ), {"email": email, "hashed_password": hashed_password})
        conn.commit()
    print(f"✅ (MySQL) 사용자 '{email}' 회원가입 시뮬레이션 성공")

# --- 5. (가짜) 로그인 함수 ---
def simulate_login(email, password):
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT id, hashed_password FROM users WHERE email = :email"
        ), {"email": email}).first()
        
        if result and pwd_context.verify(password, result.hashed_password):
            print(f"✅ (MySQL) 사용자 '{email}' 로그인 성공. 유저 ID: {result.id}")
            return result.id  # <-- ★★★ MySQL의 고유 ID 반환
        else:
            print(f"❌ (MySQL) 로그인 실패")
            return None

# --- 6. (핵심) 로그 저장 함수 ---
def save_detection_log_to_firebase(mysql_user_id, log_data):
    if mysql_user_id is None:
        print("로그인 실패. 로그 저장 안 함.")
        return

    # ★★★ MySQL ID를 Firebase에 저장하여 연동 ★★★
    log_data['user_id'] = mysql_user_id 
    
    ref = db.reference('/detection_logs')
    new_log_ref = ref.push(log_data)
    print(f"✅ (Firebase) 탐지 로그 저장 성공! (로그 ID: {new_log_ref.key})")
    print(f"    (연동된 MySQL User ID: {mysql_user_id})")

# --- 7. (실행) 전체 테스트 ---
print("--- 1. 회원가입 테스트 ---")
simulate_register("junkyu@test.com", "my_strong_password123")

print("\n--- 2. 로그인 및 로그 저장 테스트 ---")
logged_in_user_id = simulate_login("junkyu@test.com", "my_strong_password123")

# 가짜 탐지 결과
mock_log = {
    "status": "completed",
    "source_type": "file_upload",
    "model_result": {"prediction": "Deepfake", "confidence": 0.77},
    "created_at": "2025-11-13T01:10:00Z"
}

save_detection_log_to_firebase(logged_in_user_id, mock_log)