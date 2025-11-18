"""영상(Video) 관련 DB 모델 정의 파일.

- 어떤 사용자가 어떤 영상을 업로드/분석했는지 기록한다.
- 딥페이크 여부 및 confidence(신뢰도)도 함께 저장해서 나중에 조회 가능하게 한다.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.database import Base


class Video(Base):
    """videos 테이블에 해당하는 ORM 모델."""

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)

    # 영상을 업로드/분석한 사용자 (익명 사용 가능성을 위해 nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 영상이 어디서 왔는지 표시: 'upload' 또는 'youtube'
    source_type = Column(String(20))

    # youtube 링크일 경우 원본 URL 저장 (일반 업로드면 None)
    source_url = Column(String(500), nullable=True)

    # 서버 내에 저장된 실제 파일 경로
    file_path = Column(String(500), nullable=False)

    # 딥페이크 여부 (1: 딥페이크, 0: 정상, None: 아직 분석 전)
    is_deepfake = Column(Integer, nullable=True)

    # 모델이 예측한 신뢰도 (0.0 ~ 1.0 사이 값 예상)
    confidence = Column(Float, nullable=True)

    # 기록 생성 시각
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # User 모델과의 관계 설정 (user.videos 로 접근 가능)
    user = relationship("User", backref="videos")
