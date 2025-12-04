# -*- coding: utf-8 -*-

import os
import sys
import io

# 콘솔 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine
from app.models.user import User
from app.models import user as user_model
import bcrypt

# 테이블 생성
user_model.Base.metadata.create_all(bind=engine)

# 데이터베이스 연결
db = SessionLocal()

try:
    # 사용자 조회
    user = db.query(User).filter(User.email == '4comma3@naver.com').first()
    
    if user:
        print(f'사용자 찾음: {user.email}')
        
        # 새 비밀번호 해싱 - alswns0027로 변경
        new_password = 'alswns0027'
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(new_password.encode('utf-8'), salt)
        user.password_hash = hashed.decode('utf-8')
        
        db.commit()
        print(f'비밀번호 변경 완료: {new_password}')
        print(f'이제 Streamlit에서 로그인 가능!')
    else:
        print('사용자를 찾을 수 없습니다.')
        
finally:
    db.close()
