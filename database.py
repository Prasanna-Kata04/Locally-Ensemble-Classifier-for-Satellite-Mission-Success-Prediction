import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            address TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def register_user(name, mobile, email, address, password):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already exists"
        
        hashed_password = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (name, mobile, email, address, password)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, mobile, email, address, hashed_password))
        
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        return False, str(e)

def verify_user(email, password):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        return True, dict(user)
    return False, None

def get_user_by_email(email):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None
