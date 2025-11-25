"""MySQL 최신 데이터 확인"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('.env'), override=True)

engine = create_engine(os.getenv('MYSQL_URL'))
conn = engine.connect()

result = conn.execute(text('SELECT id, email, created_at FROM users ORDER BY id DESC LIMIT 5'))

print('\n🔍 MySQL users 테이블 최신 5명:')
for row in result:
    print(f'   ID: {row[0]}, Email: {row[1]}, Created: {row[2]}')

conn.close()
