"""Firebase 탐지 로그 저장 서비스.

환경 변수 또는 pydantic Settings(`config.settings`)에서
FIREBASE_CRED_PATH, FIREBASE_DB_URL 값을 읽어 초기화한다.

초기화 조건:
- 서비스 계정 키 경로가 설정되어 있고 파일이 존재할 때만 초기화
- 초기화 실패 시 (예: 파일 없음) 로그 저장은 무시(Silent Fail)

사용 예시:
    from app.services.firebase_logger import save_detection_log
    save_detection_log(user_id, {"status": "completed", ...})
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import firebase_admin  # type: ignore
    from firebase_admin import credentials, db  # type: ignore
except ImportError:  # firebase_admin 미설치 시 무시
    firebase_admin = None  # type: ignore

from app.core.config import settings

_firebase_ready = False


def _initialize_if_possible() -> None:
    global _firebase_ready
    if _firebase_ready:
        return
    if firebase_admin is None:
        return
    cred_path = settings.FIREBASE_CRED_PATH
    db_url = settings.FIREBASE_DB_URL or "https://sw-deepfake-project-default-rtdb.firebaseio.com/"
    if not cred_path or not os.path.exists(cred_path):
        return
    if not firebase_admin._apps:  # type: ignore[attr-defined]
        cred = credentials.Certificate(cred_path)  # type: ignore
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})  # type: ignore
    _firebase_ready = True


def save_detection_log(user_id: Optional[int], log: Dict[str, Any]) -> Optional[str]:
    """탐지 결과를 Firebase RTDB에 저장.

    초기화 조건이 충족되지 않으면 None 반환하며 아무 작업도 하지 않는다.
    성공 시 생성된 로그 키 문자열을 반환.
    """
    _initialize_if_possible()
    if not _firebase_ready or firebase_admin is None:
        return None
    if user_id is not None:
        log["user_id"] = user_id
    ref = db.reference("/detection_logs")  # type: ignore
    new_ref = ref.push(log)  # type: ignore
    return getattr(new_ref, "key", None)
