"""인증/인가(Auth) 관련 라우터.

- 회원가입(/auth/signup)
- 로그인(/auth/login)

⚠ 현재는 토큰(JWT) 발급 없이, 단순히 유저 정보만 반환하는 구조이다.
  추후 JWT를 추가하면, 여기서 액세스 토큰을 발급하도록 확장 가능하다.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserOut

# APIRouter를 이용해 /auth로 시작하는 엔드포인트들을 그룹화한다.
router = APIRouter(prefix="/auth", tags=["auth"])

# 비밀번호 해싱에 사용할 설정 (bcrypt 알고리즘 사용)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """평문 비밀번호를 안전하게 해시값으로 변환."""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """입력한 비밀번호와 저장된 해시값이 일치하는지 확인."""
    return pwd_context.verify(plain, hashed)


@router.post("/signup", response_model=UserOut)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    """회원가입 엔드포인트.

    - 이미 동일한 이메일이 존재하면 400 에러 반환
    - 아니면 새 User를 생성하고, 생성된 유저 정보를 반환
    """
    existed = db.query(User).filter(User.email == payload.email).first()
    if existed:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=UserOut)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """로그인 엔드포인트.

    - 이메일로 유저를 찾고
    - 비밀번호 검증 후
    - 일단은 UserOut 형태만 리턴 (JWT는 아직 미사용)
    """
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    return user
