"""딥페이크 탐지 모델을 호출하는 부분.

- 현재는 아직 실제 모델(XGBoost, LSTM, CNN)이 완료되지 않았기 때문에
  임시로 '랜덤 결과'를 반환하도록 구현해둔다.

- 나중에 팀원이 모델을 완성하면,
  이 파일 내부 함수(run_inference_on_video)를 수정해서
  실제 모델을 불러오고 추론하는 코드로 교체하면 된다.
"""

import random
from typing import Tuple


def run_inference_on_video(file_path: str) -> Tuple[int, float]:
    """주어진 영상 파일 경로에 대해 딥페이크 여부를 예측한다.

    매개변수:
        file_path: 분석할 영상 파일의 경로

    반환값:
        (is_deepfake, confidence)
        - is_deepfake: 1 (딥페이크), 0 (정상)
        - confidence: 모델의 신뢰도 (0.0 ~ 1.0 사이 값)

    현재는 아직 모델이 없으므로:
    - 0 또는 1을 랜덤으로 뽑고
    - 0.6 ~ 0.99 사이에서 랜덤 확률을 리턴한다.
    """
    is_deepfake = random.choice([0, 1])
    confidence = round(random.uniform(0.6, 0.99), 3)
    return is_deepfake, confidence
