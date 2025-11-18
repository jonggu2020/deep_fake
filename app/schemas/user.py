"""User 관련 Pydantic 스키마 정의 파일.

- 스키마(Schema)는 '요청/응답'에서 사용하는 데이터 형태를 정의한다.
- DB 모델(SQLAlchemy)과는 별도로, 외부에 노출해도 되는 필드만 선택해서 쓴다.
"""

from datetime import datetime
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """회원가입 요청 시 사용되는 데이터 형태."""

    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """로그인 요청 시 사용되는 데이터 형태."""

    email: EmailStr
    password: str


class UserOut(BaseModel):
    """클라이언트에게 응답으로 돌려줄 때 사용할 User 형태.

    - 비밀번호 해시 같은 민감 정보는 포함하지 않는다.
    """

    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        # SQLAlchemy 객체를 이 스키마로 변환할 수 있게 해주는 옵션
        orm_mode = True
