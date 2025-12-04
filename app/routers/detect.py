"""딥페이크 탐지 관련 라우터.

- /detect/upload : 파일 업로드를 통한 분석
- /detect/youtube : 유튜브 링크를 통한 분석

현재는 inference.py에서 랜덤으로 결과를 돌려주지만,
나중에 실제 모델이 완성되면 inference.py만 교체해서 쓸 수 있도록 설계했다.
"""

from pathlib import Path
from typing import Optional
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.video import Video
from app.schemas.video import DetectResult
from app.services.youtube import download_youtube_video
from app.services.inference import run_inference_on_video
from app.services.firebase_logger import save_detection_log
from app.services.landmark_extractor import create_landmark_video
from app.services.jonggu_deepfake import detect_deepfake_from_file

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
        3) 랜드마크 추출 영상 생성 (백그라운드)
        4) inference.run_inference_on_video() 호출
        5) 결과를 DB에 업데이트 후 DetectResult 형태로 반환
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

    # 3) 랜드마크 추출 영상 생성
    landmark_result = None
    try:
        print(f"🎯 랜드마크 추출 시작: {file_path}")
        landmark_result = create_landmark_video(
            input_path=str(file_path),
            output_dir="uploads/landmarks",
            max_processing_time=3.0
        )
        
        if landmark_result["success"]:
            video.landmark_video_path = landmark_result["output_path"]
            print(f"✅ 랜드마크 영상 생성 완료: {landmark_result['output_path']}")
            print(f"   - 처리 시간: {landmark_result['processing_time']}초")
            print(f"   - 처리 프레임: {landmark_result['processed_frames']}/{landmark_result['total_frames']}")
        else:
            print(f"⚠️  랜드마크 추출 실패: {landmark_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ 랜드마크 추출 중 오류: {str(e)}")
        landmark_result = {"success": False, "error": str(e)}

    # 4) 딥페이크 탐지 수행 (현재는 랜덤)
    is_deepfake, confidence = run_inference_on_video(str(file_path))

    # 5) 결과를 DB에 저장
    video.is_deepfake = is_deepfake
    video.confidence = confidence
    db.commit()
    db.refresh(video)

    # Firebase 로그 저장 (가능한 경우만)
    try:
        log_data = {
            "status": "completed",
            "source_type": video.source_type,
            "model_result": {
                "prediction": "Deepfake" if is_deepfake else "Real",
                "confidence": confidence,
            },
            "created_at": video.created_at.isoformat(),
            "video_id": video.id,
            "file_path": video.file_path,
        }
        if video.landmark_video_path:
            log_data["landmark_video_path"] = video.landmark_video_path
        save_detection_log(video.user_id, log_data)
    except Exception:
        pass

    return DetectResult(
        video_id=video.id,
        is_deepfake=is_deepfake,
        confidence=confidence,
        landmark_video_path=video.landmark_video_path,
        landmark_info={
            "processing_time": landmark_result.get("processing_time") if landmark_result else None,
            "processed_frames": landmark_result.get("processed_frames") if landmark_result else None,
            "faces_detected": landmark_result.get("faces_detected") if landmark_result else None,
        } if landmark_result and landmark_result.get("success") else None
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
        3) 랜드마크 추출 영상 생성
        4) inference.run_inference_on_video() 호출
        5) 결과를 DB에 업데이트 후 DetectResult 형태로 반환
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

    # 3) 랜드마크 추출 영상 생성
    landmark_result = None
    try:
        print(f"🎯 랜드마크 추출 시작: {file_path}")
        landmark_result = create_landmark_video(
            input_path=file_path,
            output_dir="uploads/landmarks",
            max_processing_time=3.0
        )
        
        if landmark_result["success"]:
            video.landmark_video_path = landmark_result["output_path"]
            print(f"✅ 랜드마크 영상 생성 완료: {landmark_result['output_path']}")
            print(f"   - 처리 시간: {landmark_result['processing_time']}초")
            print(f"   - 처리 프레임: {landmark_result['processed_frames']}/{landmark_result['total_frames']}")
        else:
            print(f"⚠️  랜드마크 추출 실패: {landmark_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ 랜드마크 추출 중 오류: {str(e)}")
        landmark_result = {"success": False, "error": str(e)}

    # 4) 딥페이크 탐지 수행 (현재는 랜덤)
    is_deepfake, confidence = run_inference_on_video(file_path)

    # 5) 결과를 DB에 저장
    video.is_deepfake = is_deepfake
    video.confidence = confidence
    db.commit()
    db.refresh(video)

    # Firebase 로그 저장 (가능한 경우만)
    try:
        log_data = {
            "status": "completed",
            "source_type": video.source_type,
            "model_result": {
                "prediction": "Deepfake" if is_deepfake else "Real",
                "confidence": confidence,
            },
            "created_at": video.created_at.isoformat(),
            "video_id": video.id,
            "file_path": video.file_path,
            "source_url": video.source_url,
        }
        if video.landmark_video_path:
            log_data["landmark_video_path"] = video.landmark_video_path
        save_detection_log(video.user_id, log_data)
    except Exception:
        pass

    return DetectResult(
        video_id=video.id,
        is_deepfake=is_deepfake,
        confidence=confidence,
        landmark_video_path=video.landmark_video_path,
        landmark_info={
            "processing_time": landmark_result.get("processing_time") if landmark_result else None,
            "processed_frames": landmark_result.get("processed_frames") if landmark_result else None,
            "faces_detected": landmark_result.get("faces_detected") if landmark_result else None,
        } if landmark_result and landmark_result.get("success") else None
    )


@router.get("/landmark/{video_id}")
def get_landmark_video(
    video_id: int,
    db: Session = Depends(get_db),
):
    """생성된 랜드마크 영상을 다운로드하는 엔드포인트.
    
    Args:
        video_id: 비디오 ID
    
    Returns:
        랜드마크 영상 파일
    """
    # DB에서 비디오 정보 조회
    video = db.query(Video).filter(Video.id == video_id).first()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video.landmark_video_path:
        raise HTTPException(status_code=404, detail="Landmark video not generated yet")
    
    # 파일이 실제로 존재하는지 확인
    landmark_path = Path(video.landmark_video_path)
    if not landmark_path.exists():
        raise HTTPException(status_code=404, detail="Landmark video file not found")
    
    # 파일 반환
    return FileResponse(
        path=str(landmark_path),
        media_type="video/mp4",
        filename=f"landmark_{video_id}.mp4"
    )


@router.post("/jonggu-model")
async def detect_with_jonggu_model(
    file: UploadFile = File(...),
    user_id: Optional[int] = Form(default=None),
    sensitivity_k: float = Form(default=2.0),
    db: Session = Depends(get_db),
):
    """종구님 딥페이크 탐지 모델을 사용한 분석 엔드포인트.
    
    XGBoost + RNN AE + MultiModal AE 앙상블 모델 사용
    
    요청:
        - multipart/form-data
        - file: 영상 파일 (mp4, avi, mkv, mov)
        - user_id: (선택) 사용자 ID
        - sensitivity_k: (선택) 민감도 상수 (기본값 2.0)
    
    응답:
        - fake_probability: 딥페이크 확률 (0-100%)
        - is_fake: 딥페이크 여부
        - analysis_range: 분석 대상 구간 (초)
        - input_sharpness: 입력 영상 선명도
        - sensitivity_factor: 적용된 보정 계수
    """
    import tempfile
    
    try:
        # 1) 임시 파일에 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 2) 종구님 모델로 탐지
        result = await detect_deepfake_from_file(tmp_path, sensitivity_k)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 3) DB에 기록 저장
        video = Video(
            user_id=user_id,
            source_type="upload_jonggu_model",
            source_url=None,
            file_path=tmp_path,
            is_deepfake=result['is_fake'],
            confidence=result['fake_probability'] / 100.0  # 0-1로 정규화
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        
        # 4) Firebase에 로그 저장
        try:
            log_data = {
                "status": "completed",
                "source_type": "jonggu_model",
                "model_result": {
                    "prediction": "Deepfake" if result['is_fake'] else "Real",
                    "confidence": result['fake_probability'],
                    "input_sharpness": result['input_sharpness'],
                    "sensitivity_factor": result['sensitivity_factor'],
                    "scores": result['scores']
                },
                "created_at": video.created_at.isoformat(),
                "video_id": video.id,
            }
            save_detection_log(user_id, log_data)
        except Exception:
            pass
        
        return {
            "video_id": video.id,
            "fake_probability": result['fake_probability'],
            "is_fake": result['is_fake'],
            "analysis_range": result['analysis_range'],
            "input_sharpness": result['input_sharpness'],
            "sensitivity_factor": result['sensitivity_factor'],
            "scores": result['scores']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

