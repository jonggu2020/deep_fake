# -*- coding: utf-8 -*-
"""
종구님 딥페이크 탐지 Streamlit 앱 + FastAPI 백엔드 + ngrok 통합 실행 스크립트

실행 구성:
- uvicorn 백엔드 서버 (8000)
- ngrok 터널링 (백엔드 외부 공개)
- Streamlit 프론트엔드 - 종구님 앱 (8502)
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

# ngrok 실행 파일 경로
NGROK_PATH = r"C:\ngrok\ngrok.exe"
BACKEND_PORT = 8000
FRONTEND_PORT = 8502  # 종구님 앱

def cleanup_ports():
    """사용 중인 포트 자동 정리"""
    if os.name != 'nt':
        return
    
    ports = [BACKEND_PORT, FRONTEND_PORT, 4040]
    print("🔧 포트 정리 중...")
    
    for port in ports:
        try:
            result = subprocess.run(
                f'netstat -ano | findstr ":{port}"',
                shell=True,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pid = parts[-1]
                    if pid != '0':
                        subprocess.run(f'taskkill /F /PID {pid}', 
                                     shell=True, 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
        except:
            pass
    
    print("✅ 포트 정리 완료\n")

def get_ngrok_url(max_retries=10, delay=2):
    """ngrok API로 현재 터널 URL 가져오기"""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=3)
            if response.status_code == 200:
                tunnels = response.json().get("tunnels", [])
                for tunnel in tunnels:
                    if tunnel.get("proto") == "https":
                        url = tunnel.get("public_url")
                        print(f"✅ ngrok URL 감지: {url}")
                        return url
        except Exception:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ ngrok URL 대기 중... ({i+1}/{max_retries})")
            time.sleep(delay)
    
    print("⚠️  ngrok URL을 자동으로 가져올 수 없습니다.")
    return None

def main():
    print("=" * 60)
    print("🚀 Deepfake Detection 통합 실행 스크립트")
    print("=" * 60)
    
    # 포트 자동 정리
    cleanup_ports()
    
    processes = []
    current_dir = Path(__file__).parent.absolute()
    
    try:
        # 1. uvicorn 백엔드 시작
        print("\n[1/3] 🔧 FastAPI 백엔드 시작 중...")
        backend_cmd = f'"{sys.executable}" -m uvicorn app.main:app --reload --port {BACKEND_PORT}'
        backend_process = subprocess.Popen(
            backend_cmd,
            shell=True,
            cwd=str(current_dir),
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        processes.append(("Backend", backend_process))
        print(f"✅ 백엔드 실행 중 (http://localhost:{BACKEND_PORT})")
        time.sleep(3)
        
        # 2. ngrok 터널링 시작
        print("\n[2/3] 🌐 ngrok 터널링 시작 중...")
        if not Path(NGROK_PATH).exists():
            print(f"❌ ngrok 파일을 찾을 수 없습니다: {NGROK_PATH}")
            print(f"   현재 설정: {NGROK_PATH}")
            print("   ngrok 다운로드: https://ngrok.com/download")
            print("   start.py 파일 상단의 NGROK_PATH를 수정하세요.")
            raise FileNotFoundError(NGROK_PATH)
        
        ngrok_cmd = f'"{NGROK_PATH}" http {BACKEND_PORT}'
        ngrok_process = subprocess.Popen(
            ngrok_cmd,
            shell=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        processes.append(("ngrok", ngrok_process))
        print("✅ ngrok 실행 중 (별도 창에서 확인 가능)")
        time.sleep(5)
        
        # ngrok URL 가져오기
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            print(f"\n✅ 외부 접근 URL: {ngrok_url}")
        
        # 3. Streamlit 프론트엔드 시작 (종구님 앱)
        print("\n[3/3] 🎨 Streamlit 프론트엔드 시작 중...")
        # 종구님 모델 Streamlit 앱
        streamlit_path = current_dir / "app" / "models_jonggu" / "deepfake_detector_webapp.py"
        streamlit_cmd = f'"{sys.executable}" -m streamlit run {streamlit_path} --server.port {FRONTEND_PORT}'
        streamlit_process = subprocess.Popen(
            streamlit_cmd,
            shell=True,
            cwd=str(current_dir),
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        processes.append(("Streamlit", streamlit_process))
        print(f"✅ 프론트엔드 실행 중 (http://localhost:{FRONTEND_PORT})")
        
        print("\n" + "=" * 60)
        print("✨ 모든 서버 실행 완료!")
        print("=" * 60)
        print(f"📡 백엔드 (로컬):     http://localhost:{BACKEND_PORT}")
        print(f"📡 백엔드 (외부):     {ngrok_url or '자동 감지 대기 중...'}")
        print(f"🎨 프론트엔드 (종구님): http://localhost:{FRONTEND_PORT}")
        print(f"📊 백엔드 API:        http://localhost:{BACKEND_PORT}/docs (SwaggerUI)")
        print("=" * 60)
        print("\n💡 종료하려면 Ctrl+C를 누르세요.")
        print("💡 각 서버는 별도 창에서 실행 중입니다.")
        
        # 프로세스 모니터링
        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {name} 프로세스가 종료되었습니다.")
    
    except KeyboardInterrupt:
        print("\n\n🛑 서버 종료 중...")
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"✅ {name} 종료됨")
            except Exception:
                try:
                    proc.kill()
                    print(f"⚠️  {name} 강제 종료됨")
                except Exception:
                    pass
        print("👋 모든 서버가 종료되었습니다.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print(f"   오류 타입: {type(e).__name__}")
        for name, proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        input("\n아무 키나 누르면 종료합니다...")
        sys.exit(1)

if __name__ == "__main__":
    main()
