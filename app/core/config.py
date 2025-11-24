"""프로젝트 전역 설정 파일.

- BaseSettings를 이용해 .env 파일 또는 환경변수에서 설정값을 읽어온다.
- 지금은 DB 관련 설정만 정의해두었고, 나중에 JWT 비밀키 등도 여기서 관리할 수 있다.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 프로젝트 이름 (FastAPI 문서 제목 등에 사용)
    PROJECT_NAME: str = "Deepfake Detection Backend"

    # ⚠ 현재는 SQLite를 사용하지만, 추후 MySQL로 전환할 때를 대비해서 남겨두는 값들
    DB_USER: str = "root"
    DB_PASSWORD: str = "password"
    DB_HOST: str = "127.0.0.1"
    DB_PORT: str = "3306"
    DB_NAME: str = "deepfake_db"

    # Firebase (선택적) 환경 변수
    FIREBASE_CRED_PATH: str | None = None
    FIREBASE_DB_URL: str | None = None
    
    # ffmpeg 경로 설정 (Windows에서 PATH에 없을 경우)
    FFMPEG_PATH: str | None = None

    class Config:
        # .env 파일에서 환경변수를 읽어오겠다는 의미
        env_file = ".env"
        # .env에 추가 필드가 있어도 허용 (firebase_logger.py 등에서 직접 사용)
        extra = "ignore"


# settings 객체를 import 해서 어디서든 설정값을 사용할 수 있다.
settings = Settings()
