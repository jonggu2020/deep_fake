"""라우터 패키지 초기화 파일.

- main.py에서 from app.routers import auth, detect 로 사용할 수 있도록 해준다.
- 실제 라우터 구현은 각각 auth.py, detect.py에 있다.
"""

from . import auth, detect  # noqa: F401
