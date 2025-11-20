# services/db.py
import sqlite3
from pathlib import Path
import json

DB_PATH = Path("data/app.db")


def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """초기 테이블 생성 (존재하면 무시)"""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            source_type TEXT,   -- 'upload' or 'youtube'
            source_value TEXT,  -- 파일명 또는 유튜브 URL
            result_json TEXT,   -- 백엔드 응답 전체를 JSON 문자열로 저장
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_detection_history(username: str | None,
                           source_type: str,
                           source_value: str,
                           result: dict | None):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO detection_history (username, source_type, source_value, result_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            username or "",
            source_type,
            source_value,
            json.dumps(result, ensure_ascii=False) if result is not None else None,
        ),
    )
    conn.commit()
    conn.close()


def get_detection_history(username: str | None):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, source_type, source_value, result_json, created_at
        FROM detection_history
        WHERE (? = '' OR username = ?)
        ORDER BY created_at DESC
        LIMIT 50
        """,
        (username or "", username or ""),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
