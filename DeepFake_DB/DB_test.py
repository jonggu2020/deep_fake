"""로컬 통합 테스트 스크립트 (MySQL + Firebase).

환경 변수 사용:
  MYSQL_URL                → MySQL 연결 문자열
  FIREBASE_CREDENTIALS     → 서비스 계정 키 경로 (예: secrets/firebase-service-account.json)
  FIREBASE_DATABASE_URL    → RTDB URL
  ENABLE_FIREBASE_LOG=1    → Firebase 로그 저장 활성화 플래그

주의: 실제 서비스 계정 키는 Git에 포함하지 말 것.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가 (app 모듈 import 가능하게)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
load_dotenv(project_root / ".env")

from sqlalchemy import create_engine, text
import bcrypt
from app.services.firebase_logger import save_detection_log, _initialize_if_possible  # type: ignore

MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost/firebase_db_test")
engine = create_engine(MYSQL_URL)


def init_db():
    """테스트용 users 테이블 생성 (SQLite / MySQL 호환)"""
    create_sql = None
    if "mysql" in MYSQL_URL:
        # MySQL 문법
        create_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    else:
        # SQLite 문법
        create_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL
        )
        """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()
    print("✅ users 테이블 준비 완료")


def simulate_register(email: str, password: str) -> None:
    """중복 이메일 있으면 삭제 후 새로 등록 (개발 편의용)."""
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    with engine.connect() as conn:
        # 기존 동일 이메일 제거 (중복 방지)
        conn.execute(text("DELETE FROM users WHERE email=:e"), {"e": email})
        conn.execute(text("INSERT INTO users (email, hashed_password) VALUES (:e, :h)"),
                     {"e": email, "h": hashed})
        # 새로 등록된 사용자 id 확인
        row = conn.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email}).first()
        conn.commit()
    print(f"✅ (MySQL) 사용자 '{email}' 회원가입 완료 id={row.id if row else 'UNKNOWN'}")


def simulate_login(email: str, password: str):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, hashed_password FROM users WHERE email=:e"), {"e": email}).first()
    if row and bcrypt.checkpw(password.encode('utf-8'), row.hashed_password.encode('utf-8')):
        print(f"✅ (MySQL) 로그인 성공: {row.id}")
        return row.id
    print("❌ (MySQL) 로그인 실패")
    return None


def test_log_flow():
    print("--- DB 초기화 ---")
    init_db()
    print("--- 회원가입 ---")
    simulate_register("junkyu@test.com", "my_strong_password123")
    print("--- 로그인 ---")
    uid = simulate_login("junkyu@test.com", "my_strong_password123")
    print("--- Firebase 초기화 ---")
    _initialize_if_possible()
    print("--- 로그 저장 ---")
    log_payload = {
        "status": "completed",
        "source_type": "file_upload",
        "model_result": {"prediction": "Deepfake", "confidence": 0.77},
        "created_at": "2025-11-13T01:10:00Z"
    }
    key = save_detection_log(uid, log_payload)
    if key:
        print(f"✅ Firebase 저장 완료 key={key}")
    else:
        print("⚠️ Firebase 저장 실패 또는 비활성")
    print("Done.")


if __name__ == "__main__":
    test_log_flow()