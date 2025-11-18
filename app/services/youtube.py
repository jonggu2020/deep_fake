"""유튜브 영상 다운로드 관련 기능 모음.

- 클라이언트에서 유튜브 URL을 주면, 서버에서 해당 영상을 mp4로 다운로드한다.
- 다운로드된 파일의 경로를 반환해서, 이후 추론(inference)에 사용한다.
"""

from pathlib import Path
from pytube import YouTube

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

    yt = YouTube(url)

    # 가장 해상도가 높은 progressive(mp4) 스트림 하나를 선택
    stream = (
        yt.streams
        .filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    output_path = stream.download(output_path=str(UPLOAD_DIR))
    return output_path
