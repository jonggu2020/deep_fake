"""딥페이크 탐지 관련 라우터.

- /detect/upload : 파일 업로드를 통한 분석
- /detect/youtube : 유튜브 링크를 통한 분석

현재는 inference.py에서 랜덤으로 결과를 돌려주지만,
나중에 실제 모델이 완성되면 inference.py만 교체해서 쓸 수 있도록 설계했다.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.video import Video
from app.schemas.video import DetectResult
from app.services.youtube import download_youtube_video
from app.services.inference import run_inference_on_video

router = APIRouter(prefix="/detect", tags=["detect"])

# 업로드된 파일들이 저장될 디렉토리
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DetectResult)
async def detect_from_upload(
    file: UploadFile = File(...),
    user_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
):
    """로컬에서 업로드한 영상 파일로 딥페이크 여부를 분석하는 엔드포인트.

    요청:
        - multipart/form-data 형식
        - file: 영상 파일
        - user_id: (선택) 어떤 사용자가 요청했는지 식별하기 위한 ID

    동작:
        1) 파일을 서버의 uploads 디렉토리에 저장
        2) Video 레코드를 DB에 추가
        3) inference.run_inference_on_video() 호출
        4) 결과를 DB에 업데이트 후 DetectResult 형태로 반환
    """
    # 1) 파일 저장
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2) DB에 영상 기록 생성
    video = Video(
        user_id=user_id,
        source_type="upload",
        source_url=None,
        file_path=str(file_path),
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # 3) 딥페이크 탐지 수행 (현재는 랜덤)
    is_deepfake, confidence = run_inference_on_video(str(file_path))

    # 4) 결과를 DB에 저장
    video.is_deepfake = is_deepfake
    video.confidence = confidence
    db.commit()
    db.refresh(video)

    return DetectResult(
        video_id=video.id,
        is_deepfake=is_deepfake,
        confidence=confidence,
    )


@router.post("/youtube", response_model=DetectResult)
def detect_from_youtube(
    url: str = Form(...),
    user_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
):
    """유튜브 영상 링크로 딥페이크 여부를 분석하는 엔드포인트.

    요청:
        - form 데이터로 url, 선택적으로 user_id를 받는다.

    동작:
        1) url에 해당하는 유튜브 영상을 downloads/uploads 폴더에 mp4로 저장
        2) Video 레코드를 DB에 추가
        3) inference.run_inference_on_video() 호출
        4) 결과를 DB에 업데이트 후 DetectResult 형태로 반환
    """
    try:
        # 1) 유튜브 영상 다운로드
        file_path = download_youtube_video(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Youtube download failed: {e}")

    # 2) DB에 영상 기록 생성
    video = Video(
        user_id=user_id,
        source_type="youtube",
        source_url=url,
        file_path=file_path,
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # 3) 딥페이크 탐지 수행 (현재는 랜덤)
    is_deepfake, confidence = run_inference_on_video(file_path)

    # 4) 결과를 DB에 저장
    video.is_deepfake = is_deepfake
    video.confidence = confidence
    db.commit()
    db.refresh(video)

    return DetectResult(
        video_id=video.id,
        is_deepfake=is_deepfake,
        confidence=confidence,
    )
