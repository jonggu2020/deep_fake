"""SQLite 데이터 확인"""
import sqlite3

try:
    conn = sqlite3.connect('deepfake.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\n📋 SQLite 테이블: {[t[0] for t in tables]}")
    
    if tables:
        cursor.execute("SELECT id, email, created_at FROM users ORDER BY id DESC LIMIT 5")
        users = cursor.fetchall()
        
        print(f"\n🔍 SQLite users 테이블 최신 5명:")
        for user in users:
            print(f"   ID: {user[0]}, Email: {user[1]}, Created: {user[2]}")
    
    conn.close()
except Exception as e:
    print(f"❌ SQLite 확인 실패: {e}")
