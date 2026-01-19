"""DB 연결 및 세션, Base 클래스 정의 파일.

MySQL을 기본으로 사용하며, 환경변수 로딩 순서 문제를 해결했습니다.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 🔥 중요: database.py가 import될 때 .env를 확실하게 로드
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 경로 (절대 경로)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# .env 파일 강제 로드 (override=True로 기존 환경변수도 덮어씀)
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    print(f"✅ .env 파일 로드: {ENV_FILE}", file=sys.stderr, flush=True)
else:
    print(f"⚠️  .env 파일 없음: {ENV_FILE}", file=sys.stderr, flush=True)

# MySQL URL 가져오기
MYSQL_URL = os.getenv("MYSQL_URL")

if not MYSQL_URL:
    error_msg = "❌ CRITICAL: MYSQL_URL 환경변수가 없습니다! .env 파일을 확인하세요."
    print(error_msg, file=sys.stderr, flush=True)
    raise RuntimeError(error_msg)

DATABASE_URL = MYSQL_URL
print(f"🔌 MySQL 연결: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else '(연결 정보 숨김)'}", file=sys.stderr, flush=True)

# 엔진 생성 (MySQL 전용 - SQLite fallback 제거)
# 엔진 생성 (MySQL 전용 - SQLite fallback 제거)
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True  # MySQL 연결 끊김 방지
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy Base 클래스
Base = declarative_base()


def get_db():
    """FastAPI dependency로 사용하는 DB 세션 제공 함수."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
