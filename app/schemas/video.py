"""Video 관련 Pydantic 스키마 정의 파일."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class VideoOut(BaseModel):
    """영상 정보를 응답으로 내려줄 때 사용할 스키마."""

    id: int
    source_type: str
    source_url: Optional[str]
    file_path: str
    is_deepfake: Optional[int]
    confidence: Optional[float]
    created_at: datetime

    class Config:
        orm_mode = True


class DetectResult(BaseModel):
    """딥페이크 탐지 결과를 반환하기 위한 최소 정보 스키마."""

    video_id: int
    is_deepfake: int
    confidence: float
    landmark_video_path: Optional[str] = None
    landmark_info: Optional[dict] = None
