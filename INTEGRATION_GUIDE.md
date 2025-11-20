# Integration Guide

이 문서는 팀원이 만든 프론트엔드(HOTTI), Firebase/MySQL 연계(DB_test), 기존 FastAPI 백엔드를 통합하는 절차를 정리합니다.

## 1. 브랜치 전략
- 작업 브랜치: `merge-frontend-db` (기존 JJUNI 기준 생성)
- 가져온 코드: `deepfake_web/` 디렉토리 (HOTTI)
- Firebase 연계: `app/services/firebase_logger.py` (DB_test 재구성)
- MySQL 선택적 사용: `MYSQL_URL` 환경변수 설정 시 자동 활성

## 2. 환경 변수 (.env)
참고: `.env.example` 파일 제공.
```
PROJECT_NAME=Deepfake Detection Backend
# MYSQL_URL=mysql+pymysql://user:password@host:3306/deepfake_db
FIREBASE_CRED_PATH=secrets/firebase-service-account.json
FIREBASE_DB_URL=https://sw-deepfake-project-default-rtdb.firebaseio.com/
```

## 3. 비밀/키 관리
- 실제 서비스 계정 키 JSON은 `secrets/` 디렉토리 내부 배치 (Git 추적 제외)
- CI/CD 환경에서는 Secret Manager 또는 환경변수 BASE64 인코딩 사용 가능
- 로컬 base64 활용 예:
  ```bash
  # 인코딩
  base64 -w0 secrets/firebase-service-account.json > firebase_key.b64
  # 디코딩 (CI에서)
  base64 -d firebase_key.b64 > secrets/firebase-service-account.json
  ```

## 4. Firebase 로그 저장 흐름
1. `detect.py` 엔드포인트 처리 완료 후 결과 객체 생성
2. `save_detection_log(user_id, {...})` 호출
3. 환경 충족(FIREBASE_CRED_PATH 존재) 시 `/detection_logs` RTDB push
4. 실패 시 조용히(None) 반환 → API 안정성 유지

## 5. MySQL 연동 (선택)
- `.env`에 `MYSQL_URL` 지정 시 `app/database.py`가 자동으로 해당 URL 사용
- 미지정 시 기본 SQLite(`deepfake.db`)

## 6. 설치 및 실행
```powershell
pip install -r requirements.txt
uvicorn app.main:app --reload
```
프론트엔드 (예시, deepfake_web 폴더 내부 구조에 따라):
```powershell
python deepfake_web/main.py
```

## 7. 기능 검증 체크리스트
- [ ] API `/` 헬스 응답 확인
- [ ] 회원가입/로그인 (SQLite 혹은 MySQL) 정상 동작
- [ ] `/detect/upload` 업로드 후 응답 수신 (랜덤 결과)
- [ ] `/detect/youtube` 링크 처리 (짧은 공개 영상) → 결과 수신
- [ ] Firebase 로그 생성 (`FIREBASE_CRED_PATH` 존재 시 RTDB에 항목)
- [ ] 프론트엔드에서 업로드 → 백엔드 응답/표시
- [ ] MySQL 전환 테스트 (`MYSQL_URL` 설정 후 사용자/로그 저정)

## 8. 충돌 처리 패턴
| 상황 | 해결 | 비고 |
|------|------|------|
| requirements 중복 | 상위 최신 버전 선택, 기능 테스트 | FastAPI 호환성 우선 |
| README 병합 | 두 브랜치 주요 섹션 병합, 중복 표현 제거 | 통합 가이드 링크 추가 |
| 모델 관련 대형 파일 | 제외/무시 | .gitignore 가중치 활용 |
| 비밀 키 노출 | 즉시 revert, 키 폐기 후 재발급 | 기록에서 제거 필요 |

## 9. 확장 포인트
- JWT 인증: `config.py`에 시크릿 추가 후 미들웨어/라우터 확장
- 작업 큐(Celery/RQ) 도입: 긴 처리(고해상도 영상) 비동기화
- 로깅/관찰: 구조적 JSON 로깅 + Prometheus 메트릭
- 모델 추론 교체: `run_inference_on_video` → 클래스로 추상화 후 캐싱

## 10. 재현 명령어 요약 (PowerShell)
```powershell
# 브랜치 생성
git checkout JJUNI
git pull origin JJUNI
git checkout -b merge-frontend-db

# 프론트 가져오기
git checkout origin/HOTTI -- deepfake_web
git add deepfake_web
git commit -m "Add frontend (HOTTI) directory"

# Firebase 서비스 추가 (이미 통합된 경우 생략)
# app/services/firebase_logger.py 편집 후 커밋

# 의존성 반영
pip install -r requirements.txt

# 서버 기동
uvicorn app.main:app --reload
```

## 11. 트러블슈팅
| 증상 | 원인 | 해결 |
|------|------|------|
| Firebase 로그 미생성 | 키 파일 경로 잘못됨 | 경로/권한 확인, `_firebase_ready` 상태 출력 추가 |
| MySQL 연결 실패 | URL 오타 / 포트 차단 | DB 포트/계정 재검증, 로컬 접속 테스트 |
| 업로드 실패 | 파일 크기/형식 문제 | 파일 확장자/용량 제한 로직 추가 |
| 프론트 404 | API 주소 불일치 | `backend_api.py`에서 BASE_URL 점검 |

## 12. 보안 권장 사항
- 비밀번호 해시(bcrypt) 유지, 향후 JWT + refresh 구현 권장
- 키/토큰은 절대 레포에 커밋 금지
- 업로드 파일 스캔(필요 시) 및 임시 파일 정리 cron 도입

---
궁금한 점이나 추가 확장 필요 시 문서 보강 가능합니다.
