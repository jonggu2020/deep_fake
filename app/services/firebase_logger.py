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
import sys
from typing import Any, Dict, Optional

try:
    import firebase_admin  # type: ignore
    from firebase_admin import credentials, db  # type: ignore
except ImportError:  # firebase_admin 미설치 시 무시
    firebase_admin = None  # type: ignore

_firebase_ready = False

def _get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default) if os.getenv(key) not in ("", None) else default


def _initialize_if_possible() -> None:
    """환경변수를 바탕으로 Firebase 초기화.

    조건:
    - FIREBASE_CREDENTIALS 파일 존재
    - firebase_admin 설치됨
    이미 초기화된 경우 재실행하지 않는다.
    
    ⚠️ ENABLE_FIREBASE_LOG 조건 제거: auth.py에서도 Firebase 사용해야 하므로 항상 초기화
    """
    global _firebase_ready
    if _firebase_ready:
        return
    
    if firebase_admin is None:
        print("❌ firebase_admin 패키지 미설치 → Firebase 비활성화", file=sys.stderr, flush=True)
        return

    cred_path = _get_env("FIREBASE_CREDENTIALS")
    db_url = _get_env("FIREBASE_DATABASE_URL", "https://sw-deepfake-project-default-rtdb.firebaseio.com/")
    
    if not cred_path:
        print(f"⚠️  FIREBASE_CREDENTIALS 환경변수 없음 → Firebase 비활성화", file=sys.stderr, flush=True)
        return
    
    if not os.path.exists(cred_path):
        print(f"❌ Firebase 인증 파일 없음: {cred_path}", file=sys.stderr, flush=True)
        return
    
    try:
        if not firebase_admin._apps:  # type: ignore[attr-defined]
            cred = credentials.Certificate(cred_path)  # type: ignore
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})  # type: ignore
        _firebase_ready = True
        print(f"✅ Firebase 초기화 성공: {db_url}", file=sys.stderr, flush=True)
        
        # 선택적: 초기 로그 전체 삭제 (개발 단계) 환경변수 CLEAR_FIREBASE_LOGS=1
        if _get_env("CLEAR_FIREBASE_LOGS") == "1":
            try:
                db.reference("/detection_logs").delete()  # type: ignore
                print("[Firebase] 기존 detection_logs 전체 삭제 완료")
            except Exception as e:
                print(f"[Firebase clear failed] {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ Firebase 초기화 실패: {e}", file=sys.stderr, flush=True)
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
        key = getattr(new_ref, "key", None)
        if key:
            print(_format_log_success(key, log))
        return key
    except Exception as e:
        print(f"[Firebase log failed] {e}")
        return None

def _format_log_success(key: str, log: Dict[str, Any]) -> str:
    summary = {
        "key": key,
        "user_id": log.get("user_id"),
        "status": log.get("status"),
        "source_type": log.get("source_type"),
        "prediction": (log.get("model_result") or {}).get("prediction"),
        "confidence": (log.get("model_result") or {}).get("confidence"),
    }
    # 단일 라인 요약 + 세부 JSON
    import json
    return (
        f"[Firebase log saved] key={summary['key']} user={summary['user_id']} "
        f"status={summary['status']} source={summary['source_type']} "
        f"pred={summary['prediction']} conf={summary['confidence']}\n"
        f"  full={json.dumps(log, ensure_ascii=False)}"
    )

def clear_detection_logs() -> bool:
    """수동으로 detection_logs 전체 삭제 (개발용)."""
    _initialize_if_possible()
    if not _firebase_ready or firebase_admin is None:
        return False
    try:
        db.reference("/detection_logs").delete()  # type: ignore
        print("[Firebase] detection_logs 전체 삭제 완료")
        return True
    except Exception as e:
        print(f"[Firebase clear failed] {e}")
        return False
