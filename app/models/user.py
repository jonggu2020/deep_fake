"""사용자(User) 관련 DB 모델 정의 파일.

- SQLAlchemy ORM을 사용해서 실제 DB 테이블 구조를 정의한다.
- User 테이블은 회원가입/로그인 기능 및 나중에 '내가 분석한 영상' 기록과 연결될 수 있다.
"""

from sqlalchemy import Column, Integer, String, DateTime, func
from app.database import Base


class User(Base):
    """users 테이블에 해당하는 ORM 모델."""

    __tablename__ = "users"

    # 기본 키 (PK)
    id = Column(Integer, primary_key=True, index=True)

    # 이메일: unique + index 로 설정해서 빠르게 검색 가능하도록 함
    email = Column(String(255), unique=True, index=True, nullable=False)

    # 비밀번호는 그대로 저장하지 않고, 해시값만 저장한다.
    hashed_password = Column(String(255), nullable=False)

    # 생성 시각 (서버 시간이 자동으로 들어가도록 server_default 사용)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
