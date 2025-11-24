@echo off
chcp 65001 > nul
echo ========================================
echo   Deepfake Detection 서버 시작
echo ========================================
echo.

REM conda 환경 활성화 (환경 이름 확인 필요)
call conda activate deepfake_backend_env

REM Python 스크립트 실행
python start.py

pause
