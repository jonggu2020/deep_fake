"""Firebase 탐지 로그 저장 서비스.

v5 개선 사항:
1. .env 환경변수 기반 초기화 (FIREBASE_CREDENTIALS, FIREBASE_DATABASE_URL, ENABLE_FIREBASE_LOG)
2. 설정 불충족 시 완전 무시 (Silent) → 애플리케이션 흐름 차단 없음
3. 재초기화 방지 및 안전한 단일 호출 보장
4. 실패 시 최소한의 stderr 출력으로 문제 추적

필수 환경 변수 예시 (.env):
    FIREBASE_CREDENTIALS=secrets/firebase-service-account.json
    FIREBASE_DATABASE_URL=https://sw-deepfake-project-default-rtdb.firebaseio.com/
    ENABLE_FIREBASE_LOG=1

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

_firebase_ready = False


def _initialize_if_possible() -> None:
    """환경변수를 바탕으로 Firebase 초기화.

    조건:
    - ENABLE_FIREBASE_LOG == '1'
    - FIREBASE_CREDENTIALS 파일 존재
    - firebase_admin 설치됨
    이미 초기화된 경우 재실행하지 않는다.
    """
    global _firebase_ready
    if _firebase_ready or firebase_admin is None:
        return

    if os.getenv("ENABLE_FIREBASE_LOG") != "1":
        return

    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    db_url = os.getenv("FIREBASE_DATABASE_URL", "https://sw-deepfake-project-default-rtdb.firebaseio.com/")
    if not cred_path or not os.path.exists(cred_path):
        return
    try:
        if not firebase_admin._apps:  # type: ignore[attr-defined]
            cred = credentials.Certificate(cred_path)  # type: ignore
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})  # type: ignore
        _firebase_ready = True
    except Exception as e:  # 초기화 실패 시 무시
        print(f"[Firebase init failed] {e}")
        _firebase_ready = False


def save_detection_log(user_id: Optional[int], log: Dict[str, Any]) -> Optional[str]:
    """탐지 결과를 Firebase RTDB에 저장.

    반환:
        - 성공: 새 로그 key 문자열
        - 실패/비활성: None
    """
    _initialize_if_possible()
    if not _firebase_ready or firebase_admin is None:
        return None
    if user_id is not None:
        log["user_id"] = user_id
    try:
        ref = db.reference("/detection_logs")  # type: ignore
        new_ref = ref.push(log)  # type: ignore
        return getattr(new_ref, "key", None)
    except Exception as e:
        print(f"[Firebase log failed] {e}")
        return None
