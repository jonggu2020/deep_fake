"""start.py 개선 버전 - 콘솔 출력 확인 가능"""
import subprocess
import time
import os
import sys
from pathlib import Path

def cleanup_ports():
    """포트 정리"""
    print("🔧 포트 정리 중...")
    ports = [8000, 8501, 4040]
    
    for port in ports:
        try:
            result = subprocess.run(
                f'netstat -ano | findstr ":{port}"',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        if pid.isdigit() and pid != '0':
                            subprocess.run(f'taskkill /F /PID {pid}', 
                                         shell=True, 
                                         capture_output=True)
        except:
            pass
    
    time.sleep(2)
    print("✅ 포트 정리 완료\n")

def main():
    print("=" * 60)
    print("🚀 Deepfake Detection 통합 실행 스크립트")
    print("=" * 60)
    
    cleanup_ports()
    
    # 환경변수 로드 확인
    from dotenv import load_dotenv
    env_path = Path(".env")
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ .env 파일 로드: {env_path.absolute()}\n")
    
    processes = []
    
    try:
        # 1. FastAPI 백엔드
        print("[1/3] 🔧 FastAPI 백엔드 시작 중...")
        backend = subprocess.Popen(
            ["conda", "run", "-n", "deepfake_backend_env", "--no-capture-output",
             "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',  # UTF-8 강제
            errors='replace'  # 디코딩 에러 무시
        )
        processes.append(("Backend", backend))
        time.sleep(3)
        print("✅ 백엔드 실행 중 (http://localhost:8000)\n")
        
        # 백엔드 로그 출력
        print("📋 백엔드 초기화 로그:")
        for _ in range(15):  # 처음 15줄만
            line = backend.stdout.readline()
            if line:
                print(f"   {line.strip()}")
        print()
        
        # 2. ngrok
        print("[2/3] 🌐 ngrok 터널링 시작 중...")
        ngrok = subprocess.Popen(
            ["ngrok", "http", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(("ngrok", ngrok))
        time.sleep(3)
        print("✅ ngrok 실행 중\n")
        
        # 3. Streamlit
        print("[3/3] 🎨 Streamlit 프론트엔드 시작 중...")
        streamlit = subprocess.Popen(
            ["conda", "run", "-n", "deepfake_backend_env", "--no-capture-output",
             "streamlit", "run", "deepfake_web/main.py", 
             "--server.port", "8501", "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(("Streamlit", streamlit))
        time.sleep(3)
        print("✅ 프론트엔드 실행 중 (http://localhost:8501)\n")
        
        print("=" * 60)
        print("✨ 모든 서버 실행 완료!")
        print("=" * 60)
        print("📡 백엔드:      http://localhost:8000")
        print("🌐 프론트엔드:  http://localhost:8501")
        print("=" * 60)
        print("\n💡 종료하려면 Ctrl+C를 누르세요.")
        
        # 프로세스 모니터링
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 서버 종료 중...")
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"✅ {name} 종료됨")
            except:
                proc.kill()
                print(f"✅ {name} 강제 종료됨")
        
        cleanup_ports()
        print("👋 모든 서버가 종료되었습니다.")

if __name__ == "__main__":
    main()
