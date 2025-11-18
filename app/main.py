"""FastAPI 애플리케이션 엔트리 포인트.

- uvicorn app.main:app --reload 명령으로 실행된다.
- 여기서 라우터 등록, CORS 설정, DB 테이블 생성 등을 한 번에 수행한다.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.routers import auth, detect

# 앱 실행 시점에 한 번 DB 테이블들을 생성한다.
# (이미 테이블이 있으면 그대로 사용, 없으면 새로 만든다.)
Base.metadata.create_all(bind=engine)

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="Deepfake Detection Backend")


# CORS 설정
# - 프론트엔드(React, Vue 등)에서 이 백엔드 API를 호출할 수 있도록 허용하는 설정
# - 개발 단계에서는 allow_origins=["*"] 로 두고, 운영 단계에서 도메인을 제한하는 것이 좋다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router)
app.include_router(detect.router)


@app.get("/")
def root():
    """헬스 체크용 기본 엔드포인트.

    - 서버가 잘 떠 있는지 확인할 때 사용.
    - 브라우저에서 http://127.0.0.1:8000/ 로 접속하면 이 메시지가 보인다.
    """
    return {"message": "Deepfake backend is running"}
