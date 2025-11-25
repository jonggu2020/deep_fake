"""DB 테이블 초기화 스크립트

기존 테이블을 삭제하고 백엔드 모델에 맞춰 새로 생성합니다.
"""
from app.database import engine, Base, DATABASE_URL
from app.models.user import User
from app.models.video import Video

print("\n" + "="*60)
print("🗄️  데이터베이스 초기화")
print("="*60)
print(f"DB: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}\n")

# 기존 테이블 삭제
print("⚠️  기존 테이블 삭제 중...")
Base.metadata.drop_all(bind=engine)
print("✅ 기존 테이블 삭제 완료")

# 새 테이블 생성
print("\n🔨 새 테이블 생성 중...")
Base.metadata.create_all(bind=engine)
print("✅ 테이블 생성 완료")

print("\n📋 생성된 테이블:")
print("   - users (id, email, hashed_password, created_at)")
print("   - videos (id, user_id, source_type, source_url, file_path, ...")
print("\n" + "="*60)
print("✨ 초기화 완료! 이제 백엔드를 실행하세요.")
print("="*60 + "\n")
