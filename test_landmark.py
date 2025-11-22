#!/usr/bin/env python3
"""랜드마크 추출 기능 테스트 스크립트.

이 스크립트를 사용하여 로컬에서 랜드마크 추출을 테스트할 수 있습니다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.services.landmark_extractor import create_landmark_video


def test_landmark_extraction(video_path: str):
    """랜드마크 추출 테스트.
    
    Args:
        video_path: 테스트할 영상 파일 경로
    """
    print("=" * 60)
    print("🎬 얼굴 랜드마크 추출 테스트")
    print("=" * 60)
    print()
    
    # 파일 존재 확인
    if not Path(video_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {video_path}")
        return
    
    print(f"📁 입력 파일: {video_path}")
    print()
    
    # 랜드마크 추출 시작
    print("🎯 랜드마크 추출을 시작합니다...")
    start_time = time.time()
    
    try:
        result = create_landmark_video(
            input_path=video_path,
            output_dir="uploads/landmarks",
            max_processing_time=3.0
        )
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 60)
        
        if result["success"]:
            print("✅ 랜드마크 추출 성공!")
            print()
            print("📊 처리 결과:")
            print(f"   - 출력 파일: {result['output_path']}")
            print(f"   - 처리 프레임: {result['processed_frames']}/{result['total_frames']}")
            print(f"   - 얼굴 감지: {result['faces_detected']}프레임")
            print(f"   - 해상도: {result['resolution']}")
            print(f"   - FPS: {result['fps']}")
            print(f"   - 처리 시간: {result['processing_time']}초")
            print(f"   - 전체 시간: {total_time:.2f}초")
            print()
            print("🎥 생성된 영상을 확인하세요:")
            print(f"   {result['output_path']}")
            
            # 통계
            detection_rate = (result['faces_detected'] / result['processed_frames'] * 100) if result['processed_frames'] > 0 else 0
            print()
            print("📈 통계:")
            print(f"   - 얼굴 감지율: {detection_rate:.1f}%")
            print(f"   - 초당 처리 프레임: {result['processed_frames'] / result['processing_time']:.1f} fps")
        else:
            print("❌ 랜드마크 추출 실패!")
            print(f"   오류: {result.get('error', 'Unknown error')}")
        
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 예외 발생: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


def main():
    """메인 함수."""
    if len(sys.argv) < 2:
        print("사용법: python test_landmark.py <video_file>")
        print()
        print("예시:")
        print("  python test_landmark.py test_video.mp4")
        print("  python test_landmark.py uploads/sample.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_landmark_extraction(video_path)


if __name__ == "__main__":
    main()
