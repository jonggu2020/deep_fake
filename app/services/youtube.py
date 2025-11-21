"""유튜브 영상 다운로드 관련 기능 모음.

- 클라이언트에서 유튜브 URL을 주면, 서버에서 해당 영상을 mp4로 다운로드한다.
- 다운로드된 파일의 경로를 반환해서, 이후 추론(inference)에 사용한다.
"""

import os
from pathlib import Path
import yt_dlp

# 업로드 및 다운로드 된 영상이 저장될 디렉토리
UPLOAD_DIR = Path("uploads")


def download_youtube_video(url: str) -> str:
    """유튜브 영상 URL을 받아서 로컬에 mp4 파일로 저장하고, 그 경로를 반환한다.

    매개변수:
        url: 유튜브 영상 주소

    반환값:
        다운로드된 영상 파일의 '절대 경로' 또는 '상대 경로' 문자열
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # yt-dlp 옵션 설정
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # mp4 우선, 없으면 최고 화질
        'outtmpl': str(UPLOAD_DIR / '%(id)s.%(ext)s'),  # 파일명: 비디오ID.mp4
        'quiet': True,  # 로그 최소화
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        ext = info['ext']
        output_path = str(UPLOAD_DIR / f"{video_id}.{ext}")
    
    return output_path
