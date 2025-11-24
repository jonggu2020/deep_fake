# services/backend_api.py
import requests

# 영상 다운로드 + 랜드마크 추출 등 오래 걸리는 작업 대비
TIMEOUT = 300  # 5분


def post_signup(base_url: str, username: str, password: str, email: str | None = None):
    url = base_url.rstrip("/") + "/auth/signup"
    payload = {"username": username, "password": password}
    if email:
        payload["email"] = email
    return requests.post(url, json=payload, timeout=TIMEOUT)


def post_login(base_url: str, username: str, password: str):
    url = base_url.rstrip("/") + "/auth/login"
    payload = {"username": username, "password": password}
    return requests.post(url, json=payload, timeout=TIMEOUT)


def post_detect_upload(
    base_url: str,
    access_token: str | None,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
):
    url = base_url.rstrip("/") + "/detect/upload"
    files = {
        # 백엔드에서 요구하는 필드명에 맞게 수정 (예: 'video')
        "file": (filename, file_bytes, mime_type),
    }

    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    return requests.post(url, headers=headers, files=files, timeout=TIMEOUT)


def post_detect_youtube(
    base_url: str,
    access_token: str | None,
    youtube_url: str,
):
    url = base_url.rstrip("/") + "/detect/youtube"
    data = {"url": youtube_url}  # Form 데이터로 전송

    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    return requests.post(url, headers=headers, data=data, timeout=TIMEOUT)


def get_root(base_url: str):
    url = base_url.rstrip("/") + "/"
    return requests.get(url, timeout=TIMEOUT)
