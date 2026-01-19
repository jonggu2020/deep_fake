"""인증/인가(Auth) 관련 라우터.

- 회원가입(/auth/signup)
- 로그인(/auth/login)

⚠ 현재는 토큰(JWT) 발급 없이, 단순히 유저 정보만 반환하는 구조이다.
  추후 JWT를 추가하면, 여기서 액세스 토큰을 발급하도록 확장 가능하다.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserOut

# Firebase - firebase_logger에서 초기화한 것을 재사용
import sys
try:
    from firebase_admin import db as firebase_db
    import firebase_admin
    _firebase_available = True
    
    # Firebase 초기화 확인 및 강제 초기화
    if not firebase_admin._apps:
        print("⚠️  [auth.py] Firebase 미초기화 감지 - firebase_logger import 시도...", file=sys.stderr, flush=True)
        from app.services import firebase_logger
        firebase_logger._initialize_if_possible()
        
        if firebase_admin._apps:
            print("✅ [auth.py] Firebase 초기화 성공!", file=sys.stderr, flush=True)
        else:
            print("❌ [auth.py] Firebase 초기화 실패!", file=sys.stderr, flush=True)
    else:
        print("✅ [auth.py] Firebase 이미 초기화됨", file=sys.stderr, flush=True)
        
except ImportError:
    _firebase_available = False
    firebase_admin = None
    print("❌ [auth.py] firebase_admin 패키지 없음", file=sys.stderr, flush=True)

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


def save_user_to_firebase(user: User) -> None:
    """Firebase Realtime Database에 사용자 정보 저장
    
    firebase_logger.py에서 초기화한 Firebase 인스턴스를 재사용합니다.
    """
    # 로그 파일 절대 경로
    from pathlib import Path
    LOG_FILE = Path(__file__).parent.parent.parent / "firebase_debug.log"
    
    # 로그 파일에 상세 기록
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"   [save_user_to_firebase] 시작\n")
        f.write(f"   _firebase_available: {_firebase_available}\n")
        
    if not _firebase_available:
        print(f"⚠️  [Firebase] firebase_admin 패키지 없음", file=sys.stderr, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"   ❌ firebase_admin 패키지 없음\n")
        return
    
    # firebase_admin이 초기화되었는지 확인
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"   firebase_admin: {firebase_admin}\n")
        if firebase_admin:
            f.write(f"   firebase_admin._apps: {firebase_admin._apps}\n")
    
    if firebase_admin is None or not firebase_admin._apps:
        print(f"⚠️  [Firebase] Firebase 미초기화 - 사용자 저장 건너뜀", file=sys.stderr, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"   ❌ Firebase 미초기화\n")
        return
    
    try:
        # users/{user_id} 경로에 저장
        user_ref = firebase_db.reference(f"/users/user_{user.id}")
        user_data = {
            "id": user.id,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"   Firebase 저장 시도: {user_data}\n")
        
        user_ref.set(user_data)
        print(f"✅ [Firebase] 사용자 저장 성공: user_{user.id} ({user.email})", file=sys.stderr, flush=True)
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"   ✅ Firebase 저장 성공!\n")
            
    except Exception as e:
        print(f"❌ [Firebase] 사용자 저장 실패: {e}", file=sys.stderr, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"   ❌ Firebase 저장 실패: {e}\n")
        import traceback
        traceback.print_exc()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"   Traceback:\n")
            traceback.print_exc(file=f)


@router.post("/signup", response_model=UserOut)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    """회원가입 엔드포인트.

    - 이미 동일한 이메일이 존재하면 400 에러 반환
    - 아니면 새 User를 생성하고, 생성된 유저 정보를 반환
    - Firebase Realtime Database에도 사용자 정보 저장
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
    
    # 로그 파일에 기록
    from pathlib import Path
    LOG_FILE = Path(__file__).parent.parent.parent / "firebase_debug.log"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.utcnow()}] 회원가입: user_{user.id} ({user.email})\n")
        f.write(f"   Firebase available: {_firebase_available}\n")
        if firebase_admin:
            f.write(f"   Firebase apps: {firebase_admin._apps}\n")
        f.write(f"   save_user_to_firebase 호출 전...\n")
    
    # Firebase에도 저장
    save_user_to_firebase(user)
    
    # 로그 파일에 완료 기록
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"   save_user_to_firebase 호출 완료\n")
    
    return user


@router.post("/login", response_model=UserOut)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """로그인 엔드포인트.

    - 이메일로 유저를 찾고
    - 비밀번호 검증 후
    - 일단은 UserOut 형태만 리턴 (JWT는 아직 미사용)
    - Firebase에 마지막 로그인 시간 업데이트
    """
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    
    # Firebase에 로그인 시간 업데이트
    save_user_to_firebase(user)
    
    return user
