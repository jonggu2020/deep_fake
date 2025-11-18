"""DB 연결 및 세션, Base 클래스 정의 파일.

❗ 현재 단계에서는 MySQL 서버가 준비되지 않았기 때문에,
개발 편의를 위해 **SQLite 파일 DB**를 사용한다.

- 추후 MySQL로 전환할 때는 이 파일만 수정하면 된다.
- 그 외 models, routers, services 코드는 그대로 재사용 가능하도록 구성한다.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite 파일 DB 경로 (프로젝트 루트 기준)
# 예) deepfake_backend_commented/deepfake.db 파일이 생성된다.
DATABASE_URL = "sqlite:///./deepfake.db"

# SQLite에서만 필요한 옵션(check_same_thread=False)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# DB 세션을 만들어 주는 공장(factory) 같은 것
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 모델(BaseModel 아님)에 상속해줄 SQLAlchemy Base 클래스
Base = declarative_base()


def get_db():
    """FastAPI 의 dependency로 사용하는 DB 세션 제공 함수.

    - 요청이 들어올 때마다 SessionLocal()로 세션을 하나 생성하고
    - 요청 처리가 끝나면 finally에서 세션을 닫는다.
    - routers 파일에서 Depends(get_db) 형태로 사용한다.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
